"""Convert a Whisper transcript into non-authoritative simulator candidates."""

from dataclasses import dataclass
import re
from typing import Iterable, Literal, Mapping


_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "niner": "9",
}
_PHONETIC = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()
_NUMBER = r"(?:\d+|zero|oh|one|two|three|four|five|six|seven|eight|nine|niner)"
_NUMBER_SEQUENCE = rf"{_NUMBER}(?:\s+{_NUMBER})*"
_NONZERO_DIGIT = r"(?:one|two|three|four|five|six|seven|eight|nine|niner)"
_ALTITUDE = rf"(?:{_NONZERO_DIGIT}\s+thousand(?:\s+(?:and\s+)?{_NONZERO_DIGIT}\s+hundred)?|{_NONZERO_DIGIT}\s+hundred|{_NUMBER_SEQUENCE})"
_AIRCRAFT = re.compile(r"^([A-Z]{3})([0-9]+)([A-Z]*)$")
_SPEED_DIRECTIONS: dict[str, Literal["increase", "reduce", "maintain"] | None] = {
    "increase speed to": "increase",
    "reduce speed to": "reduce",
    "maintain speed": "maintain",
    "speed": None,
}


@dataclass(frozen=True)
class DmeTermination:
    distance_nm: int
    reference: Literal["ils"] = "ils"


@dataclass(frozen=True)
class Instruction:
    action: Literal["flight_level", "altitude", "heading", "speed", "ils"]
    value: int | str | None = None
    direction: Literal["climb", "descend", "left", "right", "increase", "reduce", "maintain"] | None = None
    unit: Literal["flight_level", "feet", "degrees", "knots", "mach"] | None = None
    comparison: Literal["at_most", "at_least"] | None = None
    until: DmeTermination | None = None


@dataclass(frozen=True)
class TranscriptResult:
    utterance_id: str
    transcript: str
    status: Literal["proposed", "partial", "rejected", "ambiguous"]
    callsign: str | None
    instructions: tuple[Instruction, ...]
    reason: str | None = None
    unparsed: str | None = None


def _normalize(text: str) -> str:
    text = text.lower().replace("x-ray", "xray")
    text = re.sub(
        r"(?<![a-z0-9-])\d(?:-\d)+(?![a-z0-9-])",
        lambda match: match.group().replace("-", ""),
        text,
    )
    text = re.sub(r"\b(climb|descend|descent)\s*,\s*(?=flight\s+level\b)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def _spoken_callsigns(callsign: str, airline: str) -> set[str]:
    match = _AIRCRAFT.fullmatch(callsign)
    if not match:
        raise ValueError(f"unsupported aircraft callsign: {callsign!r}")
    _, number, suffix = match.groups()
    name = _normalize(airline)
    if not name or not re.fullmatch(r"[a-z0-9 ]+", name):
        raise ValueError(f"invalid spoken airline name for {callsign!r}")
    suffix_words = " ".join(_PHONETIC[ord(letter) - ord("A")] for letter in suffix)
    digit_words = " ".join(("zero one two three four five six seven eight niner".split())[int(n)] for n in number)
    variants = {f"{name} {number}{suffix.lower()}"}
    for spoken_number in (number, digit_words, digit_words.replace("niner", "nine")):
        variants.add(" ".join(part for part in (name, spoken_number, suffix_words) if part))
    return variants


def _number(text: str) -> int:
    pieces = text.split()
    if len(pieces) == 1 and pieces[0].isdigit():
        return int(pieces[0])
    if any(piece not in _DIGITS for piece in pieces):
        raise ValueError("use a number or a sequence of spoken digits")
    return int("".join(_DIGITS[piece] for piece in pieces))


def _altitude(text: str) -> int:
    words = text.split()
    if "thousand" in words:
        value = int(_DIGITS[words[0]]) * 1000
        if "hundred" in words:
            value += int(_DIGITS[words[-2]]) * 100
        return value
    if "hundred" in words:
        return int(_DIGITS[words[0]]) * 100
    return _number(text)


def _speed_comparison(value: str | None) -> Literal["at_most", "at_least"] | None:
    if value == "less":
        return "at_most"
    if value == "greater":
        return "at_least"
    return None


def _dme_termination(value: str | None) -> DmeTermination | None:
    if value is None:
        return None
    distance = _number(value)
    if distance <= 0:
        raise ValueError("ILS DME distance must be greater than zero")
    return DmeTermination(distance)


def _mach_value(decimal: str | None, spoken_fraction: str | None) -> str:
    if spoken_fraction is not None:
        fraction = "".join(part if part.isdigit() else _DIGITS[part] for part in spoken_fraction.split())
        value = f"0.{fraction}"
    else:
        value = decimal
        if value is None:
            raise ValueError("Mach value is missing")
        if value.startswith("."):
            value = "0" + value
    if not any(digit in "123456789" for digit in value):
        raise ValueError("Mach value must be greater than zero")
    return value


class _ParseFailure(ValueError):
    def __init__(self, message: str, instructions: tuple[Instruction, ...], unparsed: str):
        super().__init__(message)
        self.instructions = instructions
        self.unparsed = unparsed


def _instructions(text: str) -> tuple[Instruction, ...]:
    expressions = (
        ("level", re.compile(
            rf"^(climb|descend|descent)(?:\s+and\s+maintain|\s+to)?\s+(?:flight\s+level\s+|fl(?=\d)|fl\s+)({_NUMBER_SEQUENCE})(?=[\s,]|$)")),
        ("altitude", re.compile(
            rf"^(climb|descend|descent)(?:\s+and\s+maintain|\s+to)?\s+(?:altitude\s+)?({_ALTITUDE})\s*(?:feet|ft)(?=[\s,]|$)")),
        ("heading", re.compile(
            rf"^(?:(?:turn\s+)?(left|right)\s+(?:to\s+)?heading\s+|turn\s+(left|right)\s+(?:to\s+)?|fly\s+heading\s+)({_NUMBER_SEQUENCE})(?:\s+degrees)?(?=[\s,]|$)")),
        ("speed", re.compile(
            rf"^(?P<phrase>(?:reduce|increase)\s+speed\s+to|maintain\s+speed|speed)\s+"
            rf"(?P<value>{_NUMBER_SEQUENCE})(?:\s*(?:knots|kts|kt))?"
            rf"(?:\s+or\s+(?P<comparison>less|greater))?"
            rf"(?:\s+until\s+(?P<dme>{_NUMBER_SEQUENCE})\s*dme)?(?=[\s,]|$)")),
        ("mach", re.compile(
            rf"^(?:speed\s+)?mach\s+(?:(?P<decimal>(?:\d+)?\.\d+)|"
            rf"(?:point|decimal)\s+(?P<fraction>{_NUMBER_SEQUENCE}))"
            rf"(?:\s+or\s+(?P<comparison>less|greater))?"
            rf"(?:\s+until\s+(?P<dme>{_NUMBER_SEQUENCE})\s*dme)?(?=[\s,]|$)")),
        ("ils", re.compile(
            rf"^cleared\s+(?:for\s+)?ils(?:\s+approach)?(?:\s+runway\s+({_NUMBER_SEQUENCE})(?:\s+(left|right|centre|center))?)?(?=[\s,]|$)")),
    )
    remaining = text.strip()
    result = []
    seen = set()
    unparsed = remaining
    try:
        while remaining:
            unparsed = remaining.lstrip(" ,")
            remaining = unparsed
            if result and remaining.startswith("and "):
                remaining = remaining[4:].lstrip()
            if not remaining:
                raise ValueError("incomplete instruction")
            for kind, expression in expressions:
                match = expression.match(remaining)
                if not match:
                    continue
                category = "level" if kind == "altitude" else "speed" if kind == "mach" else kind
                if category in seen:
                    raise ValueError(f"multiple {kind} instructions are unsupported")
                if kind == "level":
                    value = _number(match.group(2))
                    if not 1 <= value <= 999:
                        raise ValueError("flight level must be between 1 and 999")
                    direction = "climb" if match.group(1) == "climb" else "descend"
                    instruction = Instruction("flight_level", value, direction, "flight_level")
                elif kind == "altitude":
                    value = _altitude(match.group(2))
                    if not 100 <= value <= 99900 or value % 100:
                        raise ValueError("altitude must be 100-99900 feet in whole hundreds")
                    direction = "climb" if match.group(1) == "climb" else "descend"
                    instruction = Instruction("altitude", value, direction, "feet")
                elif kind == "heading":
                    value = _number(match.group(3))
                    if not 1 <= value <= 360:
                        raise ValueError("heading must be between 1 and 360")
                    spoken_turn = match.group(1) or match.group(2)
                    turn = None if spoken_turn is None else ("left" if spoken_turn == "left" else "right")
                    instruction = Instruction("heading", value, turn, "degrees")
                elif kind == "speed":
                    value = _number(match.group("value"))
                    if not 1 <= value <= 999:
                        raise ValueError("speed must be between 1 and 999 knots")
                    instruction = Instruction(
                        "speed", value, _SPEED_DIRECTIONS[match.group("phrase")], "knots",
                        _speed_comparison(match.group("comparison")), _dme_termination(match.group("dme")),
                    )
                elif kind == "mach":
                    instruction = Instruction(
                        "speed", _mach_value(match.group("decimal"), match.group("fraction")), unit="mach",
                        comparison=_speed_comparison(match.group("comparison")),
                        until=_dme_termination(match.group("dme")),
                    )
                else:
                    runway = None
                    if match.group(1):
                        value = _number(match.group(1))
                        if not 1 <= value <= 36:
                            raise ValueError("runway must be between 01 and 36")
                        side = {"left": "L", "right": "R", "centre": "C", "center": "C", None: ""}
                        runway = f"{value:02}{side[match.group(2)]}"
                    elif re.match(r"\s+runway\b", remaining[match.end():]):
                        raise ValueError("incomplete runway")
                    instruction = Instruction("ils", runway)
                seen.add(category)
                result.append(instruction)
                remaining = remaining[match.end():].strip()
                break
            else:
                raise ValueError(f"unsupported or incomplete instruction: {remaining!r}")
        if not result:
            raise ValueError("no instruction")
    except ValueError as exc:
        raise _ParseFailure(str(exc), tuple(result), unparsed) from exc
    return tuple(result)


def interpret_transcript(
    utterance_id: str,
    transcript: str,
    aircraft: Iterable[str],
    airlines: Mapping[str, str],
    *,
    greetings: Iterable[str] = (),
) -> TranscriptResult:
    """Propose instructions for supplied live callsigns; never mutate simulator state.

    ``airlines`` maps a three-letter aircraft prefix to its spoken name.
    Exact ICAO callsigns are also recognized without an airline mapping.
    Explicitly configured greetings may appear between callsign and instructions.
    """
    normalized = _normalize(transcript)
    normalized = normalized.strip(" ,.")
    greeting_variants = set()
    for greeting in greetings:
        greeting = _normalize(greeting).strip(" ,.")
        if not greeting or not re.fullmatch(r"[a-z ]+", greeting):
            raise ValueError("greetings must contain only letters and spaces")
        greeting_variants.add(greeting)
    matches = []
    for callsign in dict.fromkeys(aircraft):
        match = _AIRCRAFT.fullmatch(callsign)
        if not match:
            raise ValueError(f"unsupported aircraft callsign: {callsign!r}")
        airline = airlines.get(match.group(1))
        aliases = _spoken_callsigns(callsign, match.group(1))
        aliases.add(callsign.lower())
        if airline is not None:
            aliases.update(_spoken_callsigns(callsign, airline))
        for alias in sorted(aliases, key=lambda value: (-len(value), value)):
            if normalized.startswith(alias) and len(normalized) > len(alias) and normalized[len(alias)] in " ,":
                matches.append((callsign, normalized[len(alias):].lstrip(" ,")))
    if not matches:
        return TranscriptResult(utterance_id, transcript, "rejected", None, (), "callsign not found in supplied aircraft")

    candidates = []
    errors = []
    for callsign, remainder in matches:
        for greeting in sorted(greeting_variants, key=lambda value: (-len(value), value)):
            if remainder == greeting:
                remainder = ""
                break
            if remainder.startswith(greeting) and remainder[len(greeting)] in " ,":
                remainder = remainder[len(greeting):].lstrip(" ,")
                break
        try:
            candidates.append((callsign, _instructions(remainder), None, None))
        except _ParseFailure as exc:
            if exc.instructions:
                candidates.append((callsign, exc.instructions, str(exc), exc.unparsed))
            else:
                errors.append((str(exc), exc.unparsed))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) > 1:
        return TranscriptResult(utterance_id, transcript, "ambiguous", None, (), "multiple aircraft or interpretations match")
    if not candidates:
        matched_callsigns = {callsign for callsign, _ in matches}
        callsign = next(iter(matched_callsigns)) if len(matched_callsigns) == 1 else None
        reason, unparsed = errors[0]
        return TranscriptResult(utterance_id, transcript, "rejected", callsign, (), reason, unparsed)
    callsign, instructions, reason, unparsed = candidates[0]
    status = "partial" if reason is not None else "proposed"
    return TranscriptResult(utterance_id, transcript, status, callsign, instructions, reason, unparsed)


def transcribe_instruction(
    worker, utterance, aircraft: Iterable[str], airlines: Mapping[str, str], *, greetings: Iterable[str] = (),
) -> TranscriptResult:
    """Transcribe one captured utterance and return an unvalidated proposal."""
    aircraft_snapshot = tuple(aircraft)
    airline_snapshot = dict(airlines)
    greeting_snapshot = tuple(greetings)
    transcript = worker.transcribe(utterance)
    return interpret_transcript(utterance.id, transcript, aircraft_snapshot, airline_snapshot, greetings=greeting_snapshot)
