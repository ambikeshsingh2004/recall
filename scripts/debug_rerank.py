"""Show reranker scores for first-person questions vs third-person notes, before and after substitution."""

from memory_core import reranker
from memory_core.retrieval import personalize

notes = ["Sam is a designer at Pixelcraft", "Sam's partner is Leah", "Sam's trip to Japan is scheduled for March"]
for q in ["Where do I work", "Which company do I work at?", "Who is my partner", "What is my job",
          "When am I going to Japan?", "What am I allergic to?"]:
    p = personalize(q, "Sam")
    print(f"{q!r:30} {[round(s, 3) for s in reranker.score(q, notes)]}  ->  {p!r:34} {[round(s, 3) for s in reranker.score(p, notes)]}")
