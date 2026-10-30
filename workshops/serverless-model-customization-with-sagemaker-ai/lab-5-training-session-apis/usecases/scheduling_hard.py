"""Meeting scheduling with per-person working hours, half-hour time zones and
calendars shared in either local time or UTC.

Same tools and reward as scheduling_agent: 1.0 for the earliest valid slot, 0.5
for a valid but later slot, 0 for an invalid slot, no booking or several bookings.
"""

from __future__ import annotations

import json
import random
from datetime import datetime

from strands import tool

from lab.agents import count_tool_calls, count_turns, make_agent, run, transcript

ZONES = {"Berlin": 1.0, "London": 0.0, "New York": -5.0, "Chicago": -6.0, "Sao Paulo": -3.0, "Mumbai": 5.5,
         "Singapore": 8.0, "Tokyo": 9.0, "Adelaide": 10.5}  # fixed UTC offsets on the requested day
HOURS = [(9 * 60, 17 * 60), (8 * 60, 16 * 60), (10 * 60, 18 * 60), (7 * 60 + 30, 15 * 60 + 30), (12 * 60, 20 * 60)]
NAMES = ["Anna", "Ben", "Chen", "Divya", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kenji", "Lara",
         "Mateo", "Nina", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara"]
DAY = datetime(2026, 11, 12)  # a Thursday
STEP = 15  # minutes

SYSTEM_PROMPT = (
    "You schedule meetings. Each person has their own working hours in their local time zone. Calendars list busy "
    "blocks either in the person's local time or in UTC, as stated in the calendar. Find the earliest start time on "
    "the requested date (in UTC, a multiple of 15 minutes) where the meeting fits inside every attendee's working "
    "hours and overlaps none of their busy blocks. Then call book_meeting once with the start time in UTC. If no "
    "slot exists that day, call book_meeting with start_utc set to 'none'."
)


def _fmt(m: int) -> str:
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def make_people(seed: int) -> dict:
    rng = random.Random(seed)
    people = {}
    for name in NAMES:
        city = rng.choice(list(ZONES))
        h0, h1 = rng.choice(HOURS)
        busy = []
        for _ in range(rng.randint(1, 4)):
            start = rng.randrange(h0, h1 - 30, 15)
            dur = rng.choice([30, 30, 45, 60, 60, 90, 120])
            busy.append((start, min(start + dur, h1)))
        busy.sort()
        off = int(ZONES[city] * 60)
        in_utc = rng.random() < 0.35 and all(0 <= a - off and b - off <= 24 * 60 for a, b in busy)  # same UTC date
        people[name] = {"city": city, "utc_offset": ZONES[city], "hours": (h0, h1), "busy": busy,
                        "calendar_in_utc": in_utc}
    return people


def valid_for(person: dict, start_utc: int, end_utc: int) -> bool:
    off = int(person["utc_offset"] * 60)
    s, e = start_utc + off, end_utc + off
    h0, h1 = person["hours"]
    if s < h0 or e > h1:
        return False
    return all(e <= b0 or s >= b1 for b0, b1 in person["busy"])


def earliest_slot(people: dict, attendees: list[str], duration: int) -> int | None:
    for start in range(0, 24 * 60, STEP):
        if all(valid_for(people[a], start, start + duration) for a in attendees):
            return start
    return None


def make_tasks(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed + 1)
    people = make_people(seed)
    tasks = []
    while len(tasks) < n:
        k = rng.choice([3, 3, 4, 4, 5])
        att = rng.sample(NAMES, k)
        dur = rng.choice([30, 45, 60, 90])
        slot = earliest_slot(people, att, dur)
        if slot is None and sum(t["expected"] is None for t in tasks) >= 0.06 * (len(tasks) + 1):
            continue  # keep "no slot" requests at about 6%
        names = ", ".join(att[:-1]) + " and " + att[-1]
        msg = f"Please book a {dur}-minute meeting with {names} on Thursday {DAY.date().isoformat()}, as early as possible."
        tasks.append({"id": f"schedh-{seed}-{len(tasks)}", "seed": seed, "attendees": att, "duration": dur,
                      "message": msg, "expected": slot})
    return tasks


_PEOPLE: dict[int, dict] = {}


def people_for(seed: int) -> dict:
    if seed not in _PEOPLE:
        _PEOPLE[seed] = make_people(seed)
    return _PEOPLE[seed]


def make_tools(people: dict, state: dict):
    @tool
    def get_calendar(person: str) -> str:
        """Get a person's city, UTC offset, working hours and busy blocks for the requested day.

        Args:
            person: First name.
        """
        p = people.get(person.strip().title())
        if p is None:
            return "unknown person"
        h0, h1 = p["hours"]
        off = int(p["utc_offset"] * 60)
        if p["calendar_in_utc"]:
            busy, tz = [f"{_fmt(a - off)}-{_fmt(b - off)}" for a, b in p["busy"]], "UTC"
        else:
            busy, tz = [f"{_fmt(a)}-{_fmt(b)}" for a, b in p["busy"]], "local"
        return json.dumps({"person": person.strip().title(), "city": p["city"], "utc_offset_hours": p["utc_offset"],
                           "working_hours_local": f"{_fmt(h0)}-{_fmt(h1)}", "busy_times_in": tz, "busy": busy})

    @tool
    def book_meeting(attendees: list[str], start_utc: str, duration_minutes: int) -> str:
        """Book the meeting. Call this exactly once.

        Args:
            attendees: First names of all attendees.
            start_utc: Start time in UTC as HH:MM on the requested date, or 'none' if no slot exists.
            duration_minutes: Meeting length in minutes.
        """
        state["bookings"].append({"start_utc": str(start_utc).strip(), "duration": int(duration_minutes)})
        return "booked"

    return [get_calendar, book_meeting]


def _parse(hhmm: str) -> int | None:
    try:
        t = datetime.strptime(hhmm.replace("Z", "").strip()[-5:], "%H:%M")
        return t.hour * 60 + t.minute
    except ValueError:
        return None


def score(task: dict, bookings: list[dict]) -> tuple[float, dict]:
    if len(bookings) != 1:
        return 0.0, {"earliest": 0.0, "valid": 0.0}
    b = bookings[0]
    people = people_for(task["seed"])
    if b["start_utc"].lower() == "none":
        ok = task["expected"] is None
        return float(ok), {"earliest": float(ok), "valid": float(ok)}
    start = _parse(b["start_utc"])
    if start is None or b["duration"] != task["duration"]:
        return 0.0, {"earliest": 0.0, "valid": 0.0}
    valid = all(valid_for(people[a], start, start + task["duration"]) for a in task["attendees"])
    earliest = valid and start == task["expected"]
    return (1.0 if earliest else 0.5 if valid else 0.0), {"earliest": float(earliest), "valid": float(valid)}


def run_agent(payload: dict, model_config: dict) -> dict:
    state = {"bookings": []}
    agent = make_agent(model_config, make_tools(people_for(payload["seed"]), state), SYSTEM_PROMPT)
    final, error = run(agent, payload["message"])
    reward, parts = score(payload, state["bookings"])
    return {"reward": reward, "booking": state["bookings"][-1] if state["bookings"] else None,
            "turns": count_turns(agent.messages),
            "metrics": {"solved": float(reward == 1.0), **parts, "tool_calls": count_tool_calls(agent.messages),
                        "bookings": len(state["bookings"]), "agent_error": float(error is not None)},
            "transcript": transcript(agent.messages), "error": error}


def load(split: str, n: int, seed: int = 0) -> list[dict]:
    seeds = [21, 22, 23, 24] if split == "train" else [201]
    per = -(-n // len(seeds))
    tasks = [t for s in seeds for t in make_tasks(s, per)]
    random.Random(seed).shuffle(tasks)
    return tasks[:n]
