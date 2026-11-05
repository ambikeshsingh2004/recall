"""History padding for the scale test (PRD §15.5).

Inserts extra sessions between a scenario's real messages: first the scenario's own
`distractor_sessions` (similar-looking but irrelevant facts, e.g. a friend's marathon),
then generic filler chat. Deterministic, so runs are comparable and cacheable.
"""

from datetime import timedelta

from evals.timeutil import parse_time

GENERIC_SESSIONS = [
    [["How do I reverse a list in Python?", "Use my_list[::-1], or my_list.reverse() to do it in place."]],
    [["Any tips to stay focused while working from home?", "Try time-blocking, a dedicated workspace and a short break every hour."]],
    [["Explain a REST API in one line.", "A REST API exposes resources over HTTP using methods like GET, POST, PUT and DELETE."]],
    [["What should I do on a rainy weekend?", "Board games, a movie marathon, or trying a new recipe are all good options."]],
    [["What's the difference between TCP and UDP?", "TCP is reliable and ordered; UDP is faster but connectionless and unreliable."]],
    [["Suggest a name for a pet cat.", "How about Miso, Pepper or Mochi?"]],
    [["How long should I steep green tea?", "About 2 to 3 minutes at around 80 degrees Celsius."]],
    [["What's a good book on habits?", "Atomic Habits by James Clear is a popular, practical choice."]],
    [["How do I center a div in CSS?", "Use display: flex; justify-content: center; align-items: center; on the parent."]],
    [["Give me a quick stretching routine.", "Neck rolls, shoulder circles, a forward fold and a hip-flexor lunge, 30 seconds each."]],
]


def pad_scenario(scenario: dict, n: int) -> dict:
    if n <= 0:
        return scenario
    times = sorted(parse_time(m["at"]) for t in scenario["threads"] for m in t["messages"])
    start, end = times[0], times[-1]
    pool = scenario.get("distractor_sessions", []) + GENERIC_SESSIONS
    step = (end - start) / (n + 1)
    extra = []
    for i in range(n):
        at = start + step * (i + 1)
        turns = pool[i % len(pool)]
        msgs = []
        for j, (user, assistant) in enumerate(turns):
            ts = (at + timedelta(minutes=j)).strftime("%Y-%m-%d %H:%M")
            msgs += [{"at": ts, "role": "user", "content": user}, {"at": ts, "role": "assistant", "content": assistant}]
        extra.append({"thread_id": f"pad-{i}", "messages": msgs})
    return scenario | {"threads": scenario["threads"] + extra}
