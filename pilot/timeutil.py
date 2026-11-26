"""Dates and times for Pilot: one storage format, readable labels, and date phrases.

Nothing in this module reads the system clock. The harness replays weeks of use
in seconds, so "now" is always the ts it passed in, and every function here takes
that moment as an argument.
"""

import calendar
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# Spelled out (instead of strftime's %a / %b) so labels never depend on the machine's locale.
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ---------------------------------------------------------------------------
# Timestamps and labels
# ---------------------------------------------------------------------------


def coerce_ts(ts: datetime | date | str) -> datetime:
    """Return a naive datetime from a datetime, a date or an ISO-8601 string.

    A timezone-aware value keeps its wall-clock time and loses the tzinfo: the
    harness sends naive local times, and comparing aware with naive datetimes
    would raise.
    """
    if isinstance(ts, datetime):
        value = ts
    elif isinstance(ts, date):
        value = datetime(ts.year, ts.month, ts.day)
    elif isinstance(ts, str):
        value = datetime.fromisoformat(ts.strip())
    else:
        raise TypeError(f"expected a datetime or an ISO string, got {type(ts).__name__}")
    return value.replace(tzinfo=None)


def iso(dt: datetime | str) -> str:
    """"YYYY-MM-DDTHH:MM:SS", the one timestamp format written to disk (it sorts as text)."""
    return coerce_ts(dt).isoformat(timespec="seconds")


def day_label(d: date | datetime | str) -> str:
    """"Mon 2 Mar 2026". The weekday is included because users talk in weekdays."""
    if isinstance(d, str):
        d = coerce_ts(d)
    return f"{_short_day(d)} {d.year}"


def dt_label(dt: datetime | str) -> str:
    """"Mon 2 Mar 2026 09:11"."""
    dt = coerce_ts(dt)
    return f"{day_label(dt)} {dt:%H:%M}"


def ago_label(then: datetime, now: datetime) -> str:
    """How long ago `then` was, in the coarse words a person would use.

    Counted in calendar days: earlier today, yesterday, 2-13 days, 2-8 weeks,
    then months of 30 days. Never says "1 weeks" or "1 months".
    """
    days = (coerce_ts(now).date() - coerce_ts(then).date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def header(now: datetime) -> str:
    """The date line at the top of every reply prompt.

    Models are unreliable at calendar arithmetic, so we state today, yesterday and
    the Monday-to-Sunday week explicitly instead of letting the model work them out.
    """
    now = coerce_ts(now)
    today = now.date()
    monday = _monday(today)
    sunday = monday + timedelta(days=6)
    # The first date only carries a year when the week crosses New Year.
    week_start = _short_day(monday) if monday.year == sunday.year else day_label(monday)
    return (
        f"Current date and time: {day_label(today)}, {now:%H:%M}. "
        f"Yesterday was {day_label(today - timedelta(days=1))}. "
        f"This week runs {week_start} – {day_label(sunday)}."
    )


# ---------------------------------------------------------------------------
# Resolving date phrases ("next Thursday") against the moment they were said
# ---------------------------------------------------------------------------


@dataclass
class When:
    """A date phrase resolved against the moment it was said."""

    phrase: str              # the words that were resolved, as written ("next Thursday")
    date: date | None        # best reading; the first day when the phrase is a range
    alt: date | None = None  # second reading when the phrase is ambiguous
    end: date | None = None  # last day, for ranges (weeks, months, quarters, early/mid/late)


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# Full names come before their abbreviations so the regex tries "march" before "mar".
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "an": 1, "a": 1,
}
# Day ranges for "early / mid / late <Month>": thirds of the month (late runs to the last day).
PART_OF_MONTH_DAYS = {"early": (1, 10), "mid": (11, 20), "late": (21, 31)}

_WEEKDAY = "(" + "|".join(WEEKDAYS) + ")"
# A weekday written in front of an explicit date ("Thursday 12 March", "Mon 16 Mar") belongs to
# that date: it is not a second date phrase, and it can settle the year (see _year_for).
# Longer forms come first, so "thurs" is tried before "thu".
_WEEKDAY_BEFORE_DATE = (
    r"(?:(" + "|".join(WEEKDAYS) + r"|mon|tues|tue|wed|thurs|thur|thu|fri|sat|sun)\.?,?\s+(?:the\s+)?)?"
)
_MONTH = "(" + "|".join(MONTHS) + ")"
_NUMBER = r"(\d{1,3}|" + "|".join(NUMBERS) + ")"
_I = re.IGNORECASE

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
# Groups: 1 weekday (optional), 2 day, 3 month, 4 year (optional).
_DAY_MONTH = re.compile(
    r"\b" + _WEEKDAY_BEFORE_DATE + r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH
    + r"\b(?:\.?,?\s+(\d{4})\b)?",
    _I,
)
# Groups: 1 weekday (optional), 2 month, 3 day, 4 year (optional).
_MONTH_DAY = re.compile(
    r"\b" + _WEEKDAY_BEFORE_DATE + _MONTH + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4})\b)?",
    _I,
)
_END_OF_MONTH = re.compile(r"\bend\s+of\s+(?:the\s+)?(?:(this|next)\s+)?month\b|\bmonth[-\s]end\b", _I)
_QUARTER = re.compile(r"\bq([1-4])\b(?:\s+(\d{4})\b)?", _I)
_PART_OF_MONTH = re.compile(r"\b(early|mid|late)[-\s]+" + _MONTH + r"\b", _I)
_DAYS_AGO = re.compile(r"\b" + _NUMBER + r"\s+(days?|weeks?)\s+ago\b", _I)
_DAYS_AHEAD = re.compile(r"\bin\s+" + _NUMBER + r"\s+(days?|weeks?)\b", _I)
_WORD_WEEKDAY = re.compile(r"\b(last|this|next)\s+" + _WEEKDAY + r"\b", _I)
_WORD_PERIOD = re.compile(r"\b(last|this|next)\s+(week|month)\b", _I)
# A weekday followed by a day number ("Tuesday 22 and Thursday 24 September", "Monday the
# 20th") is a date whose month is written elsewhere, not the coming Tuesday, so the bare
# weekday rule skips it. A time after the weekday is fine ("Friday 10am", "Monday 11:30"),
# and so is a count after a comma ("went live Friday, 3 bugs logged").
_DAY_NUMBER_AFTER = r"\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\b(?!\s*(?:am|pm|[:.]\d))"
_BARE_WEEKDAY = re.compile(r"\b" + _WEEKDAY + r"\b(?!" + _DAY_NUMBER_AFTER + ")", _I)
# Checked in this order, so "day after tomorrow" wins over the "tomorrow" inside it.
_NEAR_DAYS = (
    (re.compile(r"\b(?:the\s+)?day\s+after\s+tomorrow\b", _I), 2),
    (re.compile(r"\b(?:the\s+)?day\s+before\s+yesterday\b", _I), -2),
    (re.compile(r"\btomorrow\b", _I), 1),
    (re.compile(r"\byesterday\b", _I), -1),
)
_SPEECH_DAY = re.compile(r"\b(?:today|tonight|this\s+(?:morning|afternoon|evening))\b", _I)

# Tense cues, for readings that could be past or future: a bare weekday ("Monday"), a date
# without a year ("14 January"), a quarter without a year ("Q1"). Only the words around the
# phrase tell "the pilot kicked off Monday" (the Monday just gone) from "the demo is Monday"
# (the coming one), or "the contract ends 31 October" (said in March: this October) from "the
# pilot started 14 January" (the January just gone). The lists are the memory review's, where
# "kicked off Monday" said on a Thursday came out as the NEXT Monday and "contract ends 31
# October" said in March as LAST October, both shown as dates worked out by code (complaint 6).
# Added from the scenarios: "shipped", "ran", "took", "happened", "ended", "expired", "renewed"
# (past); "she'll", "till", "going to", "needs", "ends", "renews", "expires", "for/to <weekday>"
# ("booked for Friday", "move Thursday's call to Friday") and "is <date>" ("Renewal is 15
# December") (future). Not the noun "renewal": it is in half the messages, past ones too
# ("Maya called on Tuesday about the renewal").
PAST_CUES = re.compile(
    r"\b(?:was|were|did|had|went|met|told|called|sent|signed|started|began|kicked\s+off|launched|"
    r"happened|hinted|flagged|confirmed|agreed|asked|shipped|ran|took|ended|expired|renewed|ago)\b",
    _I,
)
FUTURE_CUES = re.compile(
    r"\b(?:by|due|until|till|before|will|next|starts?|going\s+to|needs?|deadline|"
    r"ends|ending|renews|renewing|expires|expiring)\b"
    r"|\w['’]ll\b"
    r"|\b(?:for|to)\s+" + _WEEKDAY + r"\b"
    r"|\b(?:is|are)\s+(?:on\s+)?(?:\d|" + _WEEKDAY + r"\b)",
    _I,
)


def tense_of(text: str) -> str | None:
    """"past" or "future" when text has cues of that tense only (PAST_CUES, FUTURE_CUES);
    None when it has none, or both ("she confirmed it will be Friday")."""
    past = PAST_CUES.search(text or "") is not None
    future = FUTURE_CUES.search(text or "") is not None
    if past == future:
        return None
    return "past" if past else "future"


# Each rule below yields ((start, end), When) for every match in the text, left to right.
# `tense` is tense_of() the words around the phrase; only the rules whose reading depends
# on it use it (_day_and_month, _quarter, _bare_weekday).
_Found = Iterator[tuple[tuple[int, int], When]]


def _iso_date(text: str, today: date, tense: str | None) -> _Found:
    """2026-03-14"""
    for m in _ISO_DATE.finditer(text):
        found = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if found:
            yield m.span(), When(m.group(0), found)


def _day_and_month(text: str, today: date, tense: str | None) -> _Found:
    """14 March, 14th of March, 14 Mar 2026, March 14, Mar 14th, March 14, 2026, Thursday 12 March.

    A lowercase "may" is the verb ("3 may close next year"), not the month, so it is skipped
    here, for every caller (memory review: V3 stored "3 may" as 3 May). Without a year,
    the year comes from _year_for.
    """
    for pattern, day_group, month_group in ((_DAY_MONTH, 2, 3), (_MONTH_DAY, 3, 2)):
        for m in pattern.finditer(text):
            month_word = m.group(month_group)
            if month_word == "may":
                continue
            month, day = MONTHS[month_word.lower()], int(m.group(day_group))
            if m.group(4):
                found, alt = _safe_date(int(m.group(4)), month, day), None
            else:
                found, alt = _year_for(month, day, today, tense, _weekday_number(m.group(1)))
            if found:
                yield m.span(), When(m.group(0), found, alt=alt)


def _end_of_month(text: str, today: date, tense: str | None) -> _Found:
    """end of month, end of the month, end of this month, end of next month, month-end"""
    for m in _END_OF_MONTH.finditer(text):
        offset = 1 if (m.group(1) or "").lower() == "next" else 0
        year, month = _add_months(today.year, today.month, offset)
        yield m.span(), When(m.group(0), _last_day(year, month))


def _quarter(text: str, today: date, tense: str | None) -> _Found:
    """Q1..Q4, or "Q1 2027". Without a year: this year's quarter, or next year's once this
    year's has ended ("frozen until Q1" said in November means the coming Q1; memory review),
    unless the words around it are in the past tense ("Q1 was strong")."""
    for m in _QUARTER.finditer(text):
        first_month = 3 * int(m.group(1)) - 2
        if m.group(2):
            year = int(m.group(2))
        else:
            year = today.year
            if _last_day(year, first_month + 2) < today and tense != "past":
                year += 1
        yield m.span(), When(m.group(0), date(year, first_month, 1), end=_last_day(year, first_month + 2))


def _part_of_month(text: str, today: date, tense: str | None) -> _Found:
    """early / mid / late April: the first, middle or last third of that month."""
    for m in _PART_OF_MONTH.finditer(text):
        first_day, last_day = PART_OF_MONTH_DAYS[m.group(1).lower()]
        month = MONTHS[m.group(2).lower()]
        start = _closest(month, first_day, today)
        end = start.replace(day=min(last_day, _last_day(start.year, month).day))
        yield m.span(), When(m.group(0), start, end=end)


def _near_day(text: str, today: date, tense: str | None) -> _Found:
    """(the day before) yesterday, (the day after) tomorrow"""
    for pattern, offset in _NEAR_DAYS:
        for m in pattern.finditer(text):
            yield m.span(), When(m.group(0), today + timedelta(days=offset))


def _counted_days(text: str, today: date, tense: str | None) -> _Found:
    """3 days ago, two weeks ago, in 10 days, in two weeks"""
    for pattern, sign in ((_DAYS_AGO, -1), (_DAYS_AHEAD, 1)):
        for m in pattern.finditer(text):
            count = _number(m.group(1))
            unit = 7 if m.group(2).lower().startswith("week") else 1
            yield m.span(), When(m.group(0), today + timedelta(days=sign * count * unit))


def _word_weekday(text: str, today: date, tense: str | None) -> _Found:
    """last / this / next <weekday>.

    last: the most recent one strictly before the day it was said.
    this: the one inside the current Monday-to-Sunday week.
    next: the one inside next week; when a nearer one is coming up first
          ("next Thursday" said on a Monday), that one is kept as the alternative.
    """
    for m in _WORD_WEEKDAY.finditer(text):
        word = m.group(1).lower()
        weekday = WEEKDAYS.index(m.group(2).lower())
        if word == "last":
            yield m.span(), When(m.group(0), _previous(today, weekday))
        elif word == "this":
            yield m.span(), When(m.group(0), _monday(today) + timedelta(days=weekday))
        else:
            in_next_week = _monday(today) + timedelta(days=7 + weekday)
            soonest = _upcoming(today, weekday, today_counts=False)
            yield m.span(), When(m.group(0), in_next_week, alt=soonest if soonest != in_next_week else None)


def _word_period(text: str, today: date, tense: str | None) -> _Found:
    """last / this / next week (Monday to Sunday) and last / this / next month."""
    for m in _WORD_PERIOD.finditer(text):
        offset = {"last": -1, "this": 0, "next": 1}[m.group(1).lower()]
        if m.group(2).lower() == "week":
            start = _monday(today) + timedelta(days=7 * offset)
            yield m.span(), When(m.group(0), start, end=start + timedelta(days=6))
        else:
            year, month = _add_months(today.year, today.month, offset)
            yield m.span(), When(m.group(0), date(year, month, 1), end=_last_day(year, month))


def _bare_weekday(text: str, today: date, tense: str | None) -> _Found:
    """Monday, on Monday: the coming one (today counts), with the previous one as the
    alternative; the other way round when the words around it are in the past tense
    ("the pilot kicked off Monday", said on a Thursday, is the Monday just gone).

    "every Monday" is a habit, not a date, so it is skipped.
    """
    for m in _BARE_WEEKDAY.finditer(text):
        words_before = text[: m.start()].lower().split()
        if words_before and words_before[-1] == "every":
            continue
        weekday = WEEKDAYS.index(m.group(1).lower())
        coming, previous = _upcoming(today, weekday, today_counts=True), _previous(today, weekday)
        if tense == "past":
            yield m.span(), When(m.group(0), previous, alt=coming)
        else:
            yield m.span(), When(m.group(0), coming, alt=previous)


def _speech_day(text: str, today: date, tense: str | None) -> _Found:
    """today, tonight, this morning / afternoon / evening"""
    for m in _SPEECH_DAY.finditer(text):
        yield m.span(), When(m.group(0), today)


# Tried in this order; the first rule that matches wins. Specific rules come before the
# general ones they contain: explicit dates first, "end of next month" before "next month",
# "next Friday" before "Friday". Today / tonight / this morning come last: they usually say
# when the user is speaking, not the date of what they report ("Spoke to Ravi this morning:
# go-live moved to next Friday"), so any other date phrase in the same words wins (memory
# review: the stored date and the date hint were today's). Yesterday / tomorrow stay early.
_RULES = (
    _iso_date,
    _day_and_month,
    _end_of_month,
    _quarter,
    _part_of_month,
    _near_day,
    _counted_days,
    _word_weekday,
    _word_period,
    _bare_weekday,
    _speech_day,
)


def resolve_when(phrase: str, said_at: datetime, context: str = "") -> When | None:
    """Turn a date phrase into calendar dates, relative to when it was said.

    Works on a bare phrase ("next Thursday") or inside longer text ("the pilot
    started last Friday"), case-insensitively. Returns None when no rule matches.
    Numeric dates like "3/4" are deliberately not resolved: 3 April or 4 March?

    `context` is the sentence or quote the phrase comes from (default: the phrase
    itself). Its tense decides the readings that could be past or future: a bare
    weekday, a date or a quarter without a year (see tense_of).
    """
    if not phrase or not phrase.strip():
        return None
    today = coerce_ts(said_at).date()
    tense = tense_of(context or phrase)
    for rule in _RULES:
        for _span, found in rule(phrase, today, tense):
            return found  # the first match of the first rule that has one
    return None


def find_all_when(text: str, said_at: datetime) -> list[When]:
    """Every date phrase in text, resolved, in the order they appear.

    resolve_when gives one reading, from the most specific rule. A sentence can hold
    several ("Spoke to Ravi this morning: go-live moved to next Friday"), and the date
    hints must show them all (memory review: only "this morning" reached the prompt).
    Rules claim their words in _RULES order, and a match that overlaps words already
    claimed is skipped, so "end of next month" is not also read as "next month", nor
    "next Friday" as "Friday". The tense is judged on the whole text, so pass one
    sentence at a time when the sentences differ.
    """
    if not text or not text.strip():
        return []
    today = coerce_ts(said_at).date()
    tense = tense_of(text)
    claimed: list[tuple[int, int, When]] = []
    for rule in _RULES:
        for (start, end), found in rule(text, today, tense):
            if all(end <= taken_start or start >= taken_end for taken_start, taken_end, _ in claimed):
                claimed.append((start, end, found))
    claimed.sort(key=lambda item: item[0])
    return [found for _, _, found in claimed]


def when_label(w: When) -> str:
    """Readable form of a resolved phrase.

    "Thu 12 Mar 2026", "Thu 12 Mar 2026 (or Thu 5 Mar 2026)", "week of Mon 9 Mar 2026",
    or "Wed 1 Jul 2026 – Wed 30 Sep 2026" for other ranges. Falls back to the phrase
    when there is no date. Accepts ISO date strings too, because records store dates as text.
    """
    start, alt, end = _as_date(w.date), _as_date(w.alt), _as_date(w.end)
    if start is None:
        return w.phrase
    if end is not None and end != start:
        if start.weekday() == 0 and (end - start).days == 6:
            return f"week of {day_label(start)}"
        return f"{day_label(start)} – {day_label(end)}"
    if alt is not None and alt != start:
        return f"{day_label(start)} (or {day_label(alt)})"
    return day_label(start)


# ---------------------------------------------------------------------------
# Small calendar helpers
# ---------------------------------------------------------------------------


def _short_day(d: date) -> str:
    """"Mon 2 Mar" (no year)."""
    return f"{DAY_NAMES[d.weekday()]} {d.day} {MONTH_NAMES[d.month - 1]}"


def _monday(day: date) -> date:
    """The Monday that starts the week containing `day` (weeks run Monday to Sunday)."""
    return day - timedelta(days=day.weekday())


def _previous(day: date, weekday: int) -> date:
    """The most recent `weekday` strictly before `day`."""
    back = (day.weekday() - weekday) % 7
    if back == 0:
        back = 7  # "last Monday" said on a Monday means a week ago
    return day - timedelta(days=back)


def _upcoming(day: date, weekday: int, today_counts: bool) -> date:
    """The next `weekday` on or after `day` (or strictly after it when today does not count)."""
    ahead = (weekday - day.weekday()) % 7
    if ahead == 0 and not today_counts:
        ahead = 7
    return day + timedelta(days=ahead)


def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    """(year, month) moved by `offset` months: (2026, 12) + 1 -> (2027, 1)."""
    months_since_year_zero = year * 12 + (month - 1) + offset
    return months_since_year_zero // 12, months_since_year_zero % 12 + 1


def _last_day(year: int, month: int) -> date:
    """The last calendar day of a month."""
    return date(year, month, calendar.monthrange(year, month)[1])


def _safe_date(year: int, month: int, day: int) -> date | None:
    """date(year, month, day), or None when it does not exist (31 February)."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _closest(month: int, day: int, today: date) -> date | None:
    """The date with this month and day that is nearest to `today` (last, this or next year)."""
    best = None
    for year in (today.year - 1, today.year, today.year + 1):
        candidate = _safe_date(year, month, day)
        if candidate is None:
            continue
        if best is None or abs((candidate - today).days) < abs((best - today).days):
            best = candidate
    return best


def _year_for(
    month: int, day: int, today: date, tense: str | None, weekday: int | None
) -> tuple[date | None, date | None]:
    """(date, alternative) for a day and month written without a year.

    `before` is the latest such date on or before the day it was said, `after` the earliest
    on or after it.
    1. `before` is within the last 30 days: `before`, no alternative ("28 February" said on 2 March).
    2. Future words around it ("the contract ends 31 October", "renewal is 15 December", said
       in March): `after`, with `before` as the alternative when it is the nearer one. The
       nearer date used to win whatever the words said, which put deadlines more than about
       six months ahead into last year (memory review, complaint 6).
    3. Otherwise the nearer one, as before ("the pilot started 14 January", "the 12 January
       outage", said in March or May); a past reading keeps next year's as the alternative.
       Past words do not override the distance: they are mostly reporting verbs ("Their
       sponsor confirmed the go-live is still on for 15 July", said in June, is this July).
    A weekday written with the date ("Thursday 2 April") then picks between the two readings.
    """
    years = (today.year - 1, today.year, today.year + 1)
    candidates = [found for found in (_safe_date(year, month, day) for year in years) if found]
    before = max((found for found in candidates if found <= today), default=None)
    after = min((found for found in candidates if found >= today), default=None)
    if before is None or after is None:  # only 29 February can be missing from a year
        return before or after, None
    if (today - before).days <= 30:
        return before, None
    before_is_nearer = (today - before) < (after - today)
    if tense == "future":
        best, alt = after, (before if before_is_nearer else None)
    elif before_is_nearer:
        best, alt = before, after
    else:
        best, alt = after, None
    if weekday is not None and alt is not None:
        if alt.weekday() == weekday and best.weekday() != weekday:
            return alt, None  # "Thursday 2 April" said in May: the April just gone had that Thursday
        if best.weekday() == weekday:
            return best, None
    return best, alt


def _weekday_number(word: str | None) -> int | None:
    """0 for "Monday" or "Mon" ... 6 for "Sunday" or "Sun"; None when no weekday was written."""
    if not word:
        return None
    return [name[:3] for name in WEEKDAYS].index(word[:3].lower())


def _number(word: str) -> int:
    """"3" -> 3, "three" -> 3, "a" -> 1."""
    return int(word) if word.isdigit() else NUMBERS[word.lower()]


def _as_date(value: date | datetime | str | None) -> date | None:
    """A date from a date, a datetime or an ISO string; None stays None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
