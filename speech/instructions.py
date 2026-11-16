"""Convert a Whisper transcript into non-authoritative simulator candidates."""

from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
import re
from types import MappingProxyType
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
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TRACK_DISTANCE = rf"(?:{'|'.join(_TENS)})(?:\s+{_NONZERO_DIGIT})?|(?:{'|'.join(_TEENS)})|{_NUMBER_SEQUENCE}"
_AIRCRAFT = re.compile(r"^([A-Z]{3})([0-9]+)([A-Z]*)$")
_UNMAPPED_CONTROL = re.compile(r"^([a-z]+ control)(?=[ ,]|$)")
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
class AltitudeInstruction:
    action: Literal["flight_level", "altitude"]
    value: int
    direction: Literal["climb", "descend"]
    unit: Literal["flight_level", "feet"]
    timing: Literal["now"] | None = None
    type: Literal["altitude"] = field(init=False, default="altitude")
    target_altitude_feet: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "target_altitude_feet", self.value * 100 if self.action == "flight_level" else self.value)


@dataclass(frozen=True)
class HeadingInstruction:
    value: int
    direction: Literal["left", "right"] | None = None
    type: Literal["direction"] = field(init=False, default="direction")
    action: Literal["heading"] = field(init=False, default="heading")
    unit: Literal["degrees"] = field(init=False, default="degrees")


@dataclass(frozen=True)
class DirectInstruction:
    fix: str
    type: Literal["direction"] = field(init=False, default="direction")
    action: Literal["direct"] = field(init=False, default="direct")


@dataclass(frozen=True)
class ResumeNavigationInstruction:
    type: Literal["direction"] = field(init=False, default="direction")
    action: Literal["resume_navigation"] = field(init=False, default="resume_navigation")


@dataclass(frozen=True)
class SpeedInstruction:
    value: int | str
    direction: Literal["increase", "reduce", "maintain"] | None = None
    unit: Literal["knots", "mach"] = "knots"
    comparison: Literal["or_less", "or_greater"] | None = None
    until: DmeTermination | None = None
    type: Literal["speed"] = field(init=False, default="speed")
    action: Literal["speed"] = field(init=False, default="speed")


@dataclass(frozen=True)
class IlsInstruction:
    value: str | None = None
    type: Literal["approach"] = field(init=False, default="approach")
    action: Literal["ils"] = field(init=False, default="ils")


Instruction = AltitudeInstruction | HeadingInstruction | DirectInstruction | ResumeNavigationInstruction | SpeedInstruction | IlsInstruction


@dataclass(frozen=True, init=False)
class FixCatalogue:
    """Immutable index of fix identifiers and caller-supplied spoken aliases."""

    _aliases: Mapping[str, frozenset[str]]

    def __init__(self, fixes: Mapping[str, Iterable[str]]):
        aliases: dict[str, set[str]] = {}
        for identifier, names in fixes.items():
            if not re.fullmatch(r"[A-Z0-9]{2,5}", identifier):
                raise ValueError(f"invalid fix identifier: {identifier!r}")
            if isinstance(names, str):
                raise ValueError(f"fix aliases must be an iterable of names, not a string: {identifier!r}")
            for name in (identifier, *names):
                name = _normalize(name)
                if not re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*", name):
                    raise ValueError(f"invalid spoken fix name: {name!r}")
                aliases.setdefault(name, set()).add(identifier)
        object.__setattr__(self, "_aliases", MappingProxyType({name: frozenset(ids) for name, ids in aliases.items()}))

    def lookup(self, spoken: str) -> frozenset[str]:
        return self._aliases.get(spoken, frozenset())

    def suggestions(self, spoken: str) -> frozenset[str]:
        if not spoken.isalpha() or len(spoken) < 5:
            return frozenset()
        names = (name for name in self._aliases if name.isalpha() and len(name) >= 5)
        matches = get_close_matches(spoken, names, n=5, cutoff=0.75)
        return frozenset(identifier for name in matches for identifier in self._aliases[name])


@dataclass(frozen=True)
class _AmbiguousDirect:
    fixes: frozenset[str]
    approximate: bool = False
    type: Literal["direction"] = field(init=False, default="direction")
    action: Literal["direct"] = field(init=False, default="direct")


class _AmbiguousFixError(ValueError):
    def __init__(self, fixes: frozenset[str], approximate: bool):
        self.fixes = tuple(sorted(fixes))
        prefix = "approximate fix match requires confirmation" if approximate else "multiple fixes match spoken name"
        super().__init__(prefix + ": " + ", ".join(self.fixes))


@dataclass(frozen=True)
class Identifiers:
    aircraft: str | None = None
    controller: str | None = None
    controller_spoken: str | None = None


@dataclass(frozen=True)
class Information:
    kind: Literal["track_miles"]
    value: int
    unit: Literal["nautical_miles"] = "nautical_miles"
    reference: Literal["touchdown"] = "touchdown"
    readback_required: bool = False


@dataclass(frozen=True)
class TranscriptResult:
    utterance_id: str
    transcript: str
    status: Literal["proposed", "partial", "rejected", "ambiguous"]
    identifiers: Identifiers
    instructions: tuple[Instruction, ...]
    information: tuple[Information, ...] = ()
    reason: str | None = None
    unparsed: str | None = None
    warnings: tuple[str, ...] = ()
    unrecognized: tuple[str, ...] = ()
    fix_candidates: tuple[str, ...] = ()

    @property
    def callsign(self) -> str | None:
        """Compatibility accessor; serialized results use identifiers.aircraft."""
        return self.identifiers.aircraft


def _normalize(text: str) -> str:
    text = text.lower().replace("x-ray", "xray")
    text = re.sub(
        r"(?<![a-z0-9-])\d(?:-\d)+(?![a-z0-9-])",
        lambda match: match.group().replace("-", ""),
        text,
    )
    text = re.sub(
        r"\b(climb|descend|descent)(\s+now)?\s*,\s*(?=flight\s+level\b|fl(?=\s|\d))",
        lambda match: match.group(1) + (match.group(2) or "") + " ",
        text,
    )
    text = re.sub(
        r"\bresume(?:\s*,\s*|\s+)(?:(own|main)(?:\s*,\s*|\s+))?navigation\b",
        lambda match: "resume " + (match.group(1) + " " if match.group(1) else "") + "navigation",
        text,
    )
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
    if any(piece not in _DIGITS and not (piece.isascii() and len(piece) == 1 and piece.isdigit())
           for piece in pieces):
        raise ValueError("use a number or a sequence of spoken digits")
    return int("".join(_DIGITS.get(piece, piece) for piece in pieces))


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


def _track_distance(text: str) -> int:
    parts = text.split()
    if parts[0] in _TENS:
        return _TENS[parts[0]] + (int(_DIGITS[parts[1]]) if len(parts) == 2 else 0)
    if parts[0] in _TEENS:
        return _TEENS[parts[0]]
    return _number(text)


def _speed_comparison(value: str | None) -> Literal["or_less", "or_greater"] | None:
    if value == "less":
        return "or_less"
    if value == "greater":
        return "or_greater"
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
    def __init__(self, message: str, instructions: tuple[Instruction, ...],
                 information: tuple[Information, ...], unparsed: str):
        super().__init__(message)
        self.instructions = instructions
        self.information = information
        self.unparsed = unparsed


def _contents(text: str) -> tuple[tuple[Instruction, ...], tuple[Information, ...]]:
    expressions = (
        ("level", re.compile(
            rf"^(climb|descend|descent)(?:\s+(now))?(?:\s+and\s+maintain|\s+to)?\s+(?:flight\s+level\s+|fl(?=\d)|fl\s+)({_NUMBER_SEQUENCE})(?=[\s,]|$)")),
        ("altitude", re.compile(
            rf"^(climb|descend|descent)(?:\s+(now))?(?:\s+and\s+maintain|\s+to)?\s+(?:altitude\s+)?({_ALTITUDE})\s*(?:feet|ft)(?=[\s,]|$)")),
        ("heading", re.compile(
            rf"^(?:(?:turn\s+)?(left|right)\s+(?:to\s+)?heading\s+|turn\s+(left|right)\s+(?:to\s+)?|(?:(?:fly|maintain|set|turn)\s+(?:to\s+)?)?heading\s+)({_NUMBER_SEQUENCE})(?:\s+degrees)?(?=[\s,]|$)")),
        ("resume_navigation", re.compile(r"^resume\s+(?:(?:own|main)\s+)?navigation(?=[\s,]|$)")),
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
        ("track_miles", re.compile(
            rf"^(?:you\s+have\s+)?({_TRACK_DISTANCE})\s+track\s+miles?(?=[\s,]|$)")),
    )
    remaining = text.strip()
    result = []
    information = []
    seen = set()
    unparsed = remaining
    try:
        while remaining:
            unparsed = remaining.lstrip(" ,")
            remaining = unparsed
            if (result or information) and remaining.startswith("and "):
                remaining = remaining[4:].lstrip()
            if not remaining:
                raise ValueError("incomplete instruction")
            for kind, expression in expressions:
                match = expression.match(remaining)
                if not match:
                    continue
                category = ("level" if kind == "altitude" else "speed" if kind == "mach"
                            else "direction" if kind in {"heading", "resume_navigation"} else kind)
                if kind == "track_miles":
                    information.append(Information("track_miles", _track_distance(match.group(1))))
                    remaining = remaining[match.end():].strip()
                    break
                if category in seen:
                    raise ValueError(f"multiple {kind} instructions are unsupported")
                if kind == "level":
                    value = _number(match.group(3))
                    if not 1 <= value <= 999:
                        raise ValueError("flight level must be between 1 and 999")
                    direction = "climb" if match.group(1) == "climb" else "descend"
                    instruction = AltitudeInstruction("flight_level", value, direction, "flight_level",
                                                      timing="now" if match.group(2) else None)
                elif kind == "altitude":
                    value = _altitude(match.group(3))
                    if not 100 <= value <= 99900 or value % 100:
                        raise ValueError("altitude must be 100-99900 feet in whole hundreds")
                    direction = "climb" if match.group(1) == "climb" else "descend"
                    instruction = AltitudeInstruction("altitude", value, direction, "feet",
                                                      timing="now" if match.group(2) else None)
                elif kind == "heading":
                    value = _number(match.group(3))
                    if not 1 <= value <= 360:
                        raise ValueError("heading must be between 1 and 360")
                    spoken_turn = match.group(1) or match.group(2)
                    turn = None if spoken_turn is None else ("left" if spoken_turn == "left" else "right")
                    instruction = HeadingInstruction(value, turn)
                elif kind == "resume_navigation":
                    instruction = ResumeNavigationInstruction()
                elif kind == "speed":
                    value = _number(match.group("value"))
                    if not 1 <= value <= 999:
                        raise ValueError("speed must be between 1 and 999 knots")
                    instruction = SpeedInstruction(
                        value, _SPEED_DIRECTIONS[match.group("phrase")], "knots",
                        _speed_comparison(match.group("comparison")), _dme_termination(match.group("dme")),
                    )
                elif kind == "mach":
                    instruction = SpeedInstruction(
                        _mach_value(match.group("decimal"), match.group("fraction")), unit="mach",
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
                    instruction = IlsInstruction(runway)
                seen.add(category)
                result.append(instruction)
                remaining = remaining[match.end():].strip()
                break
            else:
                raise ValueError(f"unsupported or incomplete instruction: {remaining!r}")
        if not result and not information:
            raise ValueError("no instruction or information")
    except ValueError as exc:
        raise _ParseFailure(str(exc), tuple(result), tuple(information), unparsed) from exc
    return tuple(result), tuple(information)


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


_TOKEN_PATTERN = re.compile(r"[a-z]+[0-9]+[a-z]*|[0-9]+(?:\.[0-9]+)?[a-z]*|\.[0-9]+|[a-z]+|[^\s]", re.I)
_KEYWORDS = (
    "climb", "descend", "flight", "level", "altitude", "turn", "heading",
    "speed", "knots", "mach", "cleared", "runway", "approach", "track", "miles",
)
_EXACT_WORDS = frozenset(_KEYWORDS) | frozenset(_DIGITS) | frozenset(_PHONETIC) | {
    "left", "right", "fly", "maintain", "increase", "reduce", "descent",
    "greater", "less", "until", "feet", "ft", "kts", "kt", "ils", "dme",
    "point", "decimal", "now", "and", "to", "or", "you", "have", "for",
    "thousand", "hundred", "centre", "center", "fl",
    "direct", "proceed", "route", "resume", "own", "main", "navigation",
} | frozenset(_TENS) | frozenset(_TEENS)
_WORD_ALIASES = {"climbing": "climb", "climbed": "climb", "descending": "descend",
                 "descended": "descend"}


def _correct_word(word: str) -> tuple[str, str | None]:
    if word in _WORD_ALIASES:
        return _WORD_ALIASES[word], f"interpreted {word!r} as {_WORD_ALIASES[word]!r}"
    if word in _EXACT_WORDS or not word.isalpha() or len(word) < 5:
        return word, None
    scores = sorted(((SequenceMatcher(None, word, target).ratio(), target) for target in _KEYWORDS),
                    reverse=True)
    if scores[0][0] >= 0.85 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][1], f"interpreted {word!r} as {scores[0][1]!r}"
    return word, None


def _decode_contents(text: str, fixes: FixCatalogue | None) -> tuple[tuple[Instruction, ...], tuple[Information, ...], tuple[str, ...], tuple[str, ...]]:
    """Score complete command spans, then retain unmatched spans as diagnostics."""
    tokens = [_Token(m.group(), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]
    if len(tokens) > 128:
        raise ValueError("transcript exceeds 128 tokens")
    # Each state is (score, instructions, information, skipped token indexes, warnings, categories).
    states = [(0.0, (), (), (), (), frozenset())]
    paths = {0: states}
    suggestions: dict[str, frozenset[str]] = {}
    for pos, token in enumerate(tokens):
        for score, instructions, information, skipped, warnings, categories in paths.get(pos, ()):
            punctuation = token.text in {",", "."}
            connector = token.text == "and" and bool(instructions or information) and pos + 1 < len(tokens)
            skipped_next = skipped if punctuation or connector else skipped + (pos,)
            paths.setdefault(pos + 1, []).append(
                (score - (0 if punctuation or connector else 1), instructions, information, skipped_next, warnings, categories)
            )
            if punctuation:
                continue
            corrected = []
            changes = []
            for end in range(pos, min(pos + 18, len(tokens))):
                current = tokens[end]
                if current.text in {",", "?", "!", ";", ":"}:
                    break
                word, warning = _correct_word(current.text)
                corrected.append(word)
                if warning:
                    changes.append(warning)
                # Preserve adjacency for forms such as FL150 and 160kts.
                phrase = " ".join(corrected)
                raw_phrase = " ".join(t.text for t in tokens[pos:end + 1])
                direct = re.fullmatch(r"(?:(?:proceed|route)\s+)?direct(?:\s+to)?\s+([a-z0-9 ]+)", raw_phrase)
                if (not direct and instructions and isinstance(instructions[-1], ResumeNavigationInstruction)
                        and pos > 0 and tokens[pos - 1].text == "navigation"):
                    direct = re.fullmatch(r"to\s+([a-z0-9 ]+)", raw_phrase)
                fix_matches = fixes.lookup(direct.group(1)) if direct and fixes else frozenset()
                approximate = False
                if direct and fixes and not fix_matches:
                    spoken_fix = direct.group(1)
                    if spoken_fix not in suggestions:
                        suggestions[spoken_fix] = fixes.suggestions(spoken_fix)
                    fix_matches = suggestions[spoken_fix]
                    approximate = bool(fix_matches)
                if fix_matches:
                    parsed = (DirectInstruction(next(iter(fix_matches))) if len(fix_matches) == 1 and not approximate
                              else _AmbiguousDirect(fix_matches, approximate),)
                    info = ()
                    span_changes = ()
                else:
                    try:
                        parsed, info = _contents(phrase)
                    except _ParseFailure:
                        continue
                    span_changes = tuple(changes)
                    if parsed and isinstance(parsed[0], ResumeNavigationInstruction) and phrase == "resume main navigation":
                        span_changes += ("interpreted 'resume main navigation' as 'resume own navigation'",)
                if len(parsed) + len(info) != 1:
                    continue
                instruction = parsed[0] if parsed else None
                category = (instruction.type if instruction else None)
                if category in categories:
                    if (category == "direction" and "direct_after_resume" not in categories
                            and instructions and isinstance(instructions[-1], ResumeNavigationInstruction)
                            and isinstance(instruction, (DirectInstruction, _AmbiguousDirect))):
                        category = "direct_after_resume"
                    else:
                        continue
                next_word = tokens[end + 1].text if end + 1 < len(tokens) else ""
                if instruction and (
                    next_word == "-"
                    or (next_word in _DIGITS or next_word.isdigit())
                ):
                    continue
                if instruction and (
                    (instruction.action == "speed" and next_word in {"kt", "kts", "knots", "or", "until"})
                    or (instruction.action == "ils" and next_word == "runway")
                ):
                    continue
                new_categories = categories | {category} if category else categories
                new_score = score + 10 * (end - pos + 1) - 2 * len(span_changes)
                paths.setdefault(end + 1, []).append(
                    (new_score, instructions + parsed, information + info, skipped,
                     warnings + span_changes, new_categories)
                )
        # Bound alternatives per position and category set.
        for index in range(pos + 1, min(pos + 19, len(tokens) + 1)):
            if index not in paths:
                continue
            best = {}
            for state in paths[index]:
                key = state[-1]
                if key not in best or state[0] > best[key][0]:
                    best[key] = state
            paths[index] = sorted(best.values(), key=lambda state: state[0], reverse=True)[:32]
    if not tokens:
        return (), (), (), ()
    _, instructions, information, skipped, warnings, _ = max(paths[len(tokens)], key=lambda state: state[0])
    for instruction in instructions:
        if isinstance(instruction, _AmbiguousDirect):
            raise _AmbiguousFixError(instruction.fixes, instruction.approximate)
    groups = []
    for index in skipped:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    unrecognized = tuple(text[tokens[group[0]].start:tokens[group[-1]].end] for group in groups)
    return instructions, information, unrecognized, warnings


def interpret_transcript(
    utterance_id: str,
    transcript: str,
    aircraft: Iterable[str],
    airlines: Mapping[str, str],
    *,
    controllers: Mapping[str, str] | None = None,
    greetings: Iterable[str] = (),
    fixes: FixCatalogue | None = None,
) -> TranscriptResult:
    """Propose instructions for supplied live callsigns; never mutate simulator state.

    ``airlines`` maps a three-letter aircraft prefix to its spoken name.
    Exact ICAO callsigns are also recognized without an airline mapping.
    ``controllers`` maps FSD callsigns to spoken names. These are transcript
    identifiers, not authenticated session identities.
    """
    normalized = _normalize(transcript)
    normalized = normalized.strip(" ,.")
    aircraft = tuple(aircraft)
    controller_aliases: dict[str, list[str]] = {}
    for callsign, spoken in (controllers or {}).items():
        name = _normalize(spoken)
        if not callsign or not name or not re.fullmatch(r"[a-z0-9 ]+", name):
            raise ValueError("controller callsigns and spoken names must be nonempty; spoken names use letters and spaces")
        controller_aliases.setdefault(name, []).append(callsign)
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
                matches.append((callsign, normalized[len(alias):].lstrip(" ,"), len(alias)))
    if not matches:
        # Fuzzy only the airline word; the flight number and suffix must match.
        first, separator, rest = normalized.partition(" ")
        if separator:
            for callsign in dict.fromkeys(aircraft):
                airline = airlines.get(callsign[:3])
                if not airline or " " in airline.strip():
                    continue
                score = SequenceMatcher(None, first, airline.lower()).ratio()
                if not first.isalpha() or score < 0.85:
                    continue
                for alias in _spoken_callsigns(callsign, airline):
                    _, _, suffix = alias.partition(" ")
                    if rest.startswith(suffix) and len(rest) > len(suffix) and rest[len(suffix)] in " ,":
                        matches.append((callsign, rest[len(suffix):].lstrip(" ,"),
                                        len(first) + 1 + len(suffix)))
        if not matches:
            return TranscriptResult(utterance_id, transcript, "rejected", Identifiers(), (),
                                    reason="callsign not found in supplied aircraft")
        callsign_warning = ("airline name matched approximately",)
    else:
        callsign_warning = ()

    longest = max(length for _, _, length in matches)
    matches = [(callsign, remainder) for callsign, remainder, length in matches if length == longest]
    candidates = []
    errors = []
    for callsign, remainder in matches:
        identifiers = Identifiers(aircraft=callsign)
        warnings = callsign_warning
        for name in sorted(controller_aliases, key=lambda value: (-len(value), value)):
            if remainder == name or (remainder.startswith(name) and remainder[len(name)] in " ,"):
                matching_controllers = controller_aliases[name]
                if len(matching_controllers) != 1:
                    return TranscriptResult(utterance_id, transcript, "ambiguous", Identifiers(), (),
                                            reason="multiple controllers match spoken name")
                identifiers = Identifiers(callsign, matching_controllers[0], name)
                remainder = remainder[len(name):].lstrip(" ,")
                break
        if identifiers.controller_spoken is None:
            for greeting in sorted(greeting_variants, key=lambda value: (-len(value), value)):
                if remainder == greeting:
                    remainder = ""
                    break
                if remainder.startswith(greeting) and remainder[len(greeting)] in " ,":
                    remainder = remainder[len(greeting):].lstrip(" ,")
                    break
            else:
                match = _UNMAPPED_CONTROL.match(remainder)
                if match:
                    identifiers = Identifiers(callsign, None, match.group(1))
                    remainder = remainder[match.end():].lstrip(" ,")
        if identifiers.controller_spoken is not None:
            if remainder == "hello":
                remainder = ""
            elif remainder.startswith("hello") and remainder[5] in " ,":
                remainder = remainder[5:].lstrip(" ,")
            elif re.match(r"^had\s+a\s+climb\s+(?:flight\s+level\b|fl(?=\d|\s))", remainder):
                remainder = remainder[6:].lstrip()
                warnings += ("interpreted 'had a climb' as a climb instruction; wording may refer to a past climb",)
        if fixes is None and re.search(r"\b(?:direct|resume\s+(?:(?:own|main)\s+)?navigation\s+to)\b", remainder):
            warnings += ("fix catalogue not supplied; direct-to fixes cannot be resolved",)
        try:
            instructions, information, unrecognized, corrections = _decode_contents(remainder, fixes)
        except _AmbiguousFixError as exc:
            return TranscriptResult(utterance_id, transcript, "ambiguous", identifiers, (), reason=str(exc),
                                    fix_candidates=exc.fixes)
        warnings += corrections
        if instructions or information:
            reason = "unrecognized or unsupported words" if unrecognized else None
            unparsed = " ... ".join(unrecognized) if unrecognized else None
            candidates.append((identifiers, instructions, information, reason, unparsed, warnings, unrecognized))
        else:
            errors.append((identifiers, "no recognized instruction or information", remainder, unrecognized, warnings))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) > 1:
        return TranscriptResult(utterance_id, transcript, "ambiguous", Identifiers(), (),
                                reason="multiple aircraft or interpretations match")
    if not candidates:
        matched_callsigns = {callsign for callsign, _ in matches}
        identifiers = errors[0][0] if len(matched_callsigns) == 1 else Identifiers()
        _, reason, unparsed, unrecognized, warnings = errors[0]
        return TranscriptResult(utterance_id, transcript, "rejected", identifiers, (),
                                reason=reason, unparsed=unparsed, warnings=warnings, unrecognized=unrecognized)
    identifiers, instructions, information, reason, unparsed, warnings, unrecognized = candidates[0]
    status = "partial" if reason is not None else "proposed"
    return TranscriptResult(utterance_id, transcript, status, identifiers, instructions, information, reason, unparsed,
                            warnings, unrecognized)


def transcribe_instruction(
    worker, utterance, aircraft: Iterable[str], airlines: Mapping[str, str], *,
    controllers: Mapping[str, str] | None = None, greetings: Iterable[str] = (),
    fixes: FixCatalogue | None = None,
) -> TranscriptResult:
    """Transcribe one captured utterance and return an unvalidated proposal."""
    aircraft_snapshot = tuple(aircraft)
    airline_snapshot = dict(airlines)
    controller_snapshot = dict(controllers or {})
    greeting_snapshot = tuple(greetings)
    transcript = worker.transcribe(utterance)
    return interpret_transcript(utterance.id, transcript, aircraft_snapshot, airline_snapshot,
                                controllers=controller_snapshot, greetings=greeting_snapshot, fixes=fixes)
