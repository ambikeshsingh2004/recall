"""Run the Recall eval on a Kaggle GPU (T4 x2, internet on), fully local: Ollama + Postgres/pgvector.

Pushed from the laptop with:
    kaggle kernels push -p evals/kaggle --accelerator NvidiaTeslaT4
Results land in /kaggle/working/results (download with `kaggle kernels output ambikesh91/recall-eval`).
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = "https://github.com/ambikeshsingh2004/recall.git"
# (scenario set, memory budget). Quick development run: test set at the tight budget only.
# For final numbers add ("evals/scenarios/*.json", "150") and ("evals/scenarios_test/*.json", "300").
RUNS = [
    ("evals/scenarios_test/*.json", "150"),
]
WORKERS = "4"  # scenario/mode pairs graded at the same time (Ollama serves them in parallel)
MODEL = "llama3.1:8b"   # answers and query analysis (the PRD's model)
JUDGE = "gpt-oss:20b"   # stronger judge (PRD §15.6)
# Note-taking back on llama3.1:8b: gpt-oss:20b followed the note rules worse in run v2 (52% vs 70% storing).
STORING = MODEL
WORK = "/tmp/recall"
OUT = "/kaggle/working"


def run(cmd: str | list[str], check: bool = True) -> None:
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, shell=isinstance(cmd, str), text=True)
    if check and result.returncode:
        raise SystemExit(f"command failed with exit code {result.returncode}")


def wait_for(url: str, seconds: int = 180) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except OSError:
            time.sleep(2)
    raise SystemExit(f"timed out waiting for {url}")


started = time.time()

# 1. GPUs
run("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)

# 2. Postgres 16 + pgvector from the official PostgreSQL apt repo
run("apt-get -qq update > /dev/null && apt-get -qq install -y postgresql-common zstd > /dev/null")
run("/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y > /dev/null")
run("apt-get -qq install -y postgresql-16 postgresql-16-pgvector > /dev/null")
run("service postgresql start || pg_ctlcluster 16 main start")
run([
    "su", "postgres", "-c",
    "psql -v ON_ERROR_STOP=1 -c \"CREATE USER recall WITH SUPERUSER PASSWORD 'recall'\" "
    "-c \"CREATE DATABASE recall OWNER recall\"",
])

# 3. Ollama: 8k context, keep both models loaded (one per GPU fits easily)
run("curl -fsSL https://ollama.com/install.sh | sh > /dev/null")
ollama_env = os.environ | {
    "OLLAMA_CONTEXT_LENGTH": "8192",
    "OLLAMA_KEEP_ALIVE": "24h",
    "OLLAMA_MAX_LOADED_MODELS": "2",
    "OLLAMA_NUM_PARALLEL": "4",  # answer up to 4 requests per model at once (matches WORKERS)
}
subprocess.Popen(["ollama", "serve"], env=ollama_env, stdout=open(f"{OUT}/ollama.log", "w"), stderr=subprocess.STDOUT)
wait_for("http://localhost:11434/api/version")
run(f"ollama pull {MODEL} > /dev/null 2>&1")
run(f"ollama pull {JUDGE} > /dev/null 2>&1")
run("ollama list")

# 4. Our code
run(f"git clone --depth 1 {REPO} {WORK}")
os.chdir(WORK)
run("git log --oneline -1")
run(f"{sys.executable} -m pip install -q -r requirements.txt")

# 5. Configuration (replaces .env): everything local, no rate limits
os.environ.update({
    "DATABASE_URL": "postgresql://recall:recall@localhost:5432/recall",
    "LLM_BASE_URL": "http://localhost:11434/v1",
    "LLM_API_KEY": "ollama",
    "LLM_MODEL": MODEL,
    "LLM_SMALL_MODEL": MODEL,
    "LLM_REASONING_EFFORT": "",  # llama is not a reasoning model
    "LLM_JSON_MODE": "json_object",
    "LLM_TIMEOUT_S": "600",
    "JUDGE_MODEL": JUDGE,
    "STORING_MODEL": STORING,
    "MODEL_CACHE_DIR": "/tmp/models",
    "LOG_DIR": f"{OUT}/logs",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_API_KEY": "unused",
})

# 6. Quick check that the stack works before the long run
run([
    sys.executable, "-c",
    "from memory_core import db, llm; db.init_schema(); "
    "print('LLM says:', llm.complete([{'role': 'user', 'content': 'Reply with OK'}], max_tokens=20))",
])

# 7. The evals: same design as on the laptop (3 modes, memory budget). Storing calls are cached in
#    Postgres, so the second test-set run only pays for answers and judging.
for scenarios, budget in RUNS:
    run([
        sys.executable, "-m", "evals.run",
        "--modes", "full,rag,ours",
        "--scenarios", scenarios,
        "--memory-budget", budget,
        "--workers", WORKERS,
    ], check=False)

# 8. Save results where Kaggle keeps them
shutil.copytree(f"{WORK}/evals/results", f"{OUT}/results", dirs_exist_ok=True)
print(f"\nDone in {(time.time() - started) / 60:.0f} min. Results in {OUT}/results")
