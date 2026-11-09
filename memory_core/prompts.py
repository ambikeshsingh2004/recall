"""Prompt templates (PRD Appendix A). Literal braces are doubled for str.format."""

EXTRACT = """You extract long-term memories about the user from a conversation.
Today: {weekday} {date}.
{user_name_line}
Already known about the user:
{known}
Known people, projects and things (use these exact names):
{entities}

Note types:
- fact: something true now that may change later (job, tools, plans, people, preferences, upcoming dates).
- event: something that happened at a specific time, including suggestions the assistant made.
- instruction: how the user wants the assistant to behave.

Rules:
- One self-contained sentence per note. Never combine two topics in one note:
  "Arjun trains for a marathon and runs 4 days a week" must be two notes.
  Relationships are their own note: "My partner Leah and I are going to Japan in December" gives
  "Leah is Sam's partner" AND "Sam and Leah are going to Japan in December".
- Every note must make sense on its own months later: include who and what.
  A corrected or changed value is written in full ("Sam and Leah's Japan trip is in March"), never "Japan is in March".
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
- entities: exact names of other people, projects, products, places and codes in the note. Never the user themself.
- aliases: when the user refers to a person or thing by a role or nickname, add it,
  e.g. {{"name": "Rahul", "alias": "my cofounder"}}.
- forget: if the user asks to forget or delete something, add a short description of it
  (e.g. "Sam's phone number"). Do NOT create any note containing that information, and do NOT create
  notes about the forget request itself.
- corrections: if the user says something they said earlier was wrong, add a short description of the wrong
  statement (e.g. "Sam is allergic to peanuts"), and add a note with the correct information if given.
- If nothing is worth remembering, return {{"notes": [], "aliases": [], "forget": [], "corrections": []}}.

EARLIER CONTEXT:
{context}

NEW MESSAGES:
{new_messages}"""

RECONCILE = """For each NEW note, decide which OLD notes it replaces, meaning the OLD note is no longer true
because of the NEW note. OLD notes that are still true alongside the NEW note are NOT replaced
(for example, a new hobby does not replace an old hobby; a new deadline does replace the old deadline for the same thing).
A NEW note describing a change ("the release date changed to Dec 9", "moved to March") replaces the OLD note that
states the previous value of that same thing. A NEW event that does not change an OLD fact replaces nothing.

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

ANALYZE = """Today is {weekday} {today}. The user's name is {user_name}.
Recent chat:
{recent}

New message: {message}

Split the message into separate questions only if it asks for different pieces of information (max 3).
For each question return:
- search_query: a standalone search query with pronouns and references resolved, written in the third person
  with the user's name instead of I/my/me (e.g. "Where does Sam work?")
- entities: exact names of people, projects, products or codes it mentions (resolve "my cofounder" etc. if the chat says who)
- time_from, time_to: YYYY-MM-DD range if it is about a time period, else null.
  "last week" = the previous Monday to Sunday; "yesterday" = one day; "this month" = 1st of this month to today.
- needs_history: true if it asks about the past or about something that changed ("before", "used to", "originally")
Return {{"queries": [...]}}"""

SCRUB = """Rewrite this conversation summary so it no longer contains any information about: {what}
Keep everything else exactly as it is. Return only the summary.

{summary}"""

ANSWER_SYSTEM = """You are a helpful personal assistant with long-term memory of this user.
Today is {today}.
Use the memory below to answer. If memories disagree, trust the most recent one; notes marked OUTDATED describe
the past, not the present. If the answer is not in the memory or the conversation, say you don't know instead
of guessing. Follow the user's stated preferences on how to respond.

{memory}"""
