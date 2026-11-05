"""Prompt templates (PRD Appendix A). Literal braces are doubled for str.format."""

EXTRACT = """You extract long-term memories about the user from a conversation.
Today: {weekday} {date}.
Already known about the user:
{known}

Note types:
- fact: something true now that may change later (job, tools, plans, people, preferences, upcoming dates).
- event: something that happened at a specific time, including suggestions the assistant made.
- instruction: how the user wants the assistant to behave.

Rules:
- One self-contained sentence per note. Never combine two topics in one note:
  "Arjun trains for a marathon and runs 4 days a week" must be two notes.
- Refer to the user by their name whenever it appears above or in the conversation; use "The user" only if
  the name is truly unknown. Resolve all pronouns.
- Keep who said, wanted or suggested something. A wish or request is not a decision:
  "Rahul wants a free tier" must NOT become "the company offers a free tier".
- Facts about other people (friends, family, other teams) must name that person and never be written as facts
  about the user.
- Convert relative dates ("this week", "yesterday") to absolute dates using today's date.
- When something changes, write BOTH a fact with the new current state and an event describing the change.
- Keep exact names, numbers and codes (for example PG-1042).
- Skip greetings and small talk. Extract only from NEW MESSAGES; EARLIER CONTEXT is only for understanding.
- importance 0-1: how useful this is for future conversations. confidence 0-1: lower for uncertain statements.
- occurred_at: YYYY-MM-DD for events, otherwise null.
- If nothing is worth remembering, return {{"notes": []}}.

EARLIER CONTEXT:
{context}

NEW MESSAGES:
{new_messages}"""

RECONCILE = """For each NEW note, decide which OLD notes it replaces, meaning the OLD note is no longer true
because of the NEW note. OLD notes that are still true alongside the NEW note are NOT replaced
(for example, a new hobby does not replace an old hobby; a new deadline does replace the old deadline for the same thing).

{pairs}

Return {{"decisions": [{{"new": 0, "replaces": ["0a"]}}]}} with one entry per NEW note (empty list if it replaces nothing)."""

SUMMARY = """Current summary of this conversation:
{summary}

New messages:
{messages}

Update the summary. Keep earlier points short, add new points prefixed with their date (e.g. "Oct 29: ..."),
keep exact names, numbers and codes, and note anything unresolved. Return only the summary."""

PROFILE = """Write a profile of the user in under {max_words} words from the current facts below.
All facts describe the same person, the user; "The user" and the user's name refer to the same person.
Use ONLY information stated in the facts. Never invent tools, people, plans, numbers or relationships.
Cover, where the facts say so: who they are, projects, tools, key people and upcoming dates; skip anything not stated.
Write compact plain-text notes for an assistant. Return only the profile.

FACTS:
{facts}"""

ANSWER_SYSTEM = """You are a helpful personal assistant with long-term memory of this user.
Today is {today}.
Use the memory below to answer. If memories disagree, trust the most recent one; notes marked OUTDATED describe
the past, not the present. If the answer is not in the memory or the conversation, say you don't know instead
of guessing. Follow the user's stated preferences on how to respond.

{memory}"""
