from dataclasses import asdict
import json
import os
from pathlib import Path
import unittest

from speech.instructions import (
    AltitudeInstruction, DirectInstruction, DmeTermination, FixCatalogue, HeadingInstruction,
    Identifiers, IlsInstruction, Information, ResumeNavigationInstruction, SpeedInstruction,
    interpret_transcript, transcribe_instruction,
)
from speech.stt import Utterance, WhisperWorker, load_wav


def Instruction(action, value=None, direction=None, unit=None, comparison=None, until=None, timing=None):
    if action in {"flight_level", "altitude"}:
        return AltitudeInstruction(action, value, direction, unit, timing)
    if action == "heading":
        return HeadingInstruction(value, direction)
    if action == "speed":
        return SpeedInstruction(value, direction, unit, comparison, until)
    if action == "ils":
        return IlsInstruction(value)
    raise ValueError(f"unsupported test instruction: {action}")


class InstructionTests(unittest.TestCase):
    def parse(self, transcript, aircraft=("RYR12T",), airlines=None):
        return interpret_transcript("job-1", transcript, aircraft, airlines or {"RYR": "Ryanair"})

    def test_multiple_instructions_are_a_single_proposal(self):
        text = "Ryanair one two tango, turn right heading zero niner zero, descend flight level one two zero and speed 210 knots."
        result = self.parse(text)
        self.assertEqual((result.utterance_id, result.transcript, result.status, result.callsign),
                         ("job-1", text, "proposed", "RYR12T"))
        self.assertEqual(result.instructions, (
            Instruction("heading", 90, "right", "degrees"),
            Instruction("flight_level", 120, "descend", "flight_level"),
            Instruction("speed", 210, unit="knots"),
        ))

    def test_controller_identifier_and_now_are_separate_from_instructions(self):
        text = "RYR12T london control hello, climb now flight level 110, fly heading 080 degrees"
        result = interpret_transcript("job", text, ["RYR12T"], {},
                                      controllers={"LTC_N_CTR": "London Control"})
        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.identifiers, Identifiers("RYR12T", "LTC_N_CTR", "london control"))
        self.assertEqual(result.information, ())
        self.assertEqual(result.instructions, (
            Instruction("flight_level", 110, "climb", "flight_level", timing="now"),
            Instruction("heading", 80, None, "degrees"),
        ))
        self.assertEqual(asdict(result)["instructions"][0]["target_altitude_feet"], 11000)
        self.assertEqual(result.transcript, text)

    def test_instruction_serialization_contains_only_relevant_fields(self):
        result = self.parse("RYR12T fly heading 080, climb now FL110, speed 220 knots or less")
        self.assertEqual(asdict(result)["instructions"], (
            {"type": "direction", "action": "heading", "value": 80, "direction": None, "unit": "degrees"},
            {"type": "altitude", "action": "flight_level", "value": 110, "direction": "climb",
             "unit": "flight_level", "timing": "now", "target_altitude_feet": 11000},
            {"type": "speed", "action": "speed", "value": 220, "direction": None, "unit": "knots",
             "comparison": "or_less", "until": None},
        ))

    def test_direct_and_resume_navigation(self):
        fixes = FixCatalogue({"LOGAN": (), "XIDIL": (), "LAM": ("Lamborne",), "BNN": ("Bovingdon",)})
        for phrase, fix in (("direct LOGAN", "LOGAN"), ("direct to xidil", "XIDIL"),
                            ("proceed direct to lamborne", "LAM"), ("direct bovingdon", "BNN")):
            with self.subTest(phrase=phrase):
                result = interpret_transcript("job", f"RYR12T {phrase}, climb FL110", ["RYR12T"], {}, fixes=fixes)
                self.assertEqual(result.status, "proposed")
                self.assertEqual(result.instructions, (
                    DirectInstruction(fix), AltitudeInstruction("flight_level", 110, "climb", "flight_level"),
                ))
                self.assertEqual(asdict(result)["instructions"][0],
                                 {"type": "direction", "action": "direct", "fix": fix})
        result = self.parse("RYR12T resume own navigation")
        self.assertEqual((result.status, result.instructions), ("proposed", (ResumeNavigationInstruction(),)))
        self.assertEqual(asdict(result)["instructions"][0], {"type": "direction", "action": "resume_navigation"})

    def test_fix_matching_does_not_guess_unknown_or_ambiguous_names(self):
        fixes = FixCatalogue({"LAM": ("shared",), "BNN": ("shared",), "LOGAN": ()})
        result = interpret_transcript("job", "RYR12T direct to shared, climb FL110", ["RYR12T"], {}, fixes=fixes)
        self.assertEqual((result.status, result.instructions), ("ambiguous", ()))
        self.assertIn("BNN, LAM", result.reason)
        result = interpret_transcript("job", "RYR12T direct to zyzzzz, climb FL110", ["RYR12T"], {}, fixes=fixes)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.instructions, (AltitudeInstruction("flight_level", 110, "climb", "flight_level"),))
        self.assertEqual(result.unrecognized, ("direct to zyzzzz",))
        result = interpret_transcript("job", "RYR12T direct to LOGAN", ["RYR12T"], {})
        self.assertEqual((result.status, result.instructions), ("rejected", ()))
        with self.assertRaises(ValueError):
            FixCatalogue({"LAM": "Lamborne"})

    def test_test_catalogue_and_reported_navigation_phrases(self):
        path = Path(__file__).resolve().parents[1] / "speech" / "test_fixes.json"
        with path.open(encoding="utf-8") as file:
            fixes = FixCatalogue(json.load(file))
        text = "Ryanair 12T, London, hello resume main navigation direct Lambourne"
        result = interpret_transcript("job", text, ["RYR12T"], {"RYR": "Ryanair"},
                                      controllers={"LTC_N_CTR": "London"}, fixes=fixes)
        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.instructions, (ResumeNavigationInstruction(), DirectInstruction("LAM")))
        self.assertTrue(any("resume main navigation" in warning for warning in result.warnings))
        for phrase in ("resume own navigation direct to Lambourne", "resume own navigation to Lambourne"):
            with self.subTest(phrase=phrase):
                result = interpret_transcript("job", f"RYR12T {phrase}", ["RYR12T"], {}, fixes=fixes)
                self.assertEqual((result.status, result.instructions),
                                 ("proposed", (ResumeNavigationInstruction(), DirectInstruction("LAM"))))
        result = interpret_transcript("job", "RYR12T route direct to Bovingdon", ["RYR12T"], {}, fixes=fixes)
        self.assertEqual((result.status, result.instructions), ("proposed", (DirectInstruction("BNN"),)))

    def test_approximate_fix_matches_require_confirmation(self):
        fixes = FixCatalogue({"LAM": ("Lambourne",), "LMB": ("Landborne",)})
        exact = interpret_transcript("job", "RYR12T direct Landborne", ["RYR12T"], {}, fixes=fixes)
        self.assertEqual((exact.status, exact.instructions), ("proposed", (DirectInstruction("LMB"),)))
        fixes = FixCatalogue({"LAM": ("Lambourne",), "LMB": ("Landbourne",)})
        for phrase in ("direct landborne", "resume own navigation to landborne"):
            with self.subTest(phrase=phrase):
                result = interpret_transcript("job", f"RYR12T {phrase}", ["RYR12T"], {}, fixes=fixes)
                self.assertEqual((result.status, result.instructions), ("ambiguous", ()))
                self.assertEqual(result.fix_candidates, ("LAM", "LMB"))
                self.assertIn("requires confirmation", result.reason)
        result = interpret_transcript("job", "RYR12T direct LOGEN", ["RYR12T"], {},
                                      fixes=FixCatalogue({"LOGAN": ()}))
        self.assertEqual((result.status, result.instructions, result.fix_candidates), ("ambiguous", (), ("LOGAN",)))

    def test_reported_navigation_punctuation_and_missing_fix_catalogue(self):
        text = "Ryanair 12 tango resume, own navigation, direct landborne, climb, flight level 370."
        result = self.parse(text)
        self.assertEqual((result.status, result.instructions), (
            "partial", (ResumeNavigationInstruction(), AltitudeInstruction("flight_level", 370, "climb", "flight_level")),
        ))
        self.assertEqual(result.unrecognized, ("direct landborne",))
        self.assertIn("fix catalogue not supplied; direct-to fixes cannot be resolved", result.warnings)
        path = Path(__file__).resolve().parents[1] / "speech" / "test_fixes.json"
        with path.open(encoding="utf-8") as file:
            fixes = FixCatalogue(json.load(file))
        result = interpret_transcript("job", text, ["RYR12T"], {"RYR": "Ryanair"}, fixes=fixes)
        self.assertEqual((result.status, result.instructions, result.fix_candidates), ("ambiguous", (), ("LAM",)))
        result = interpret_transcript("job", text.replace("landborne", "Lambourne"),
                                      ["RYR12T"], {"RYR": "Ryanair"}, fixes=fixes)
        self.assertEqual((result.status, result.instructions), (
            "proposed", (ResumeNavigationInstruction(), DirectInstruction("LAM"),
                         AltitudeInstruction("flight_level", 370, "climb", "flight_level")),
        ))
        result = self.parse("RYR12T resume, turn heading 090")
        self.assertEqual((result.status, result.unrecognized), ("partial", ("resume",)))

    def test_fix_catalogue_is_reusable_and_snapshotted(self):
        aliases = ["Lamborne"]
        fixes = FixCatalogue({"LAM": aliases})
        aliases.clear()

        class Worker:
            def transcribe(self, utterance):
                return "RYR12T direct Lamborne"

        result = transcribe_instruction(Worker(), Utterance("job", b"\x01\x00", "", 0),
                                        ["RYR12T"], {}, fixes=fixes)
        self.assertEqual((result.status, result.instructions), ("proposed", (DirectInstruction("LAM"),)))

    def test_track_miles_are_information_not_a_readback_instruction(self):
        text = "EZY332 turn right heading 090 degrees, descend to altitude 5000ft, you have 25 track miles"
        result = interpret_transcript("job", text, ["EZY332"], {})
        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.identifiers.aircraft, "EZY332")
        self.assertEqual(result.instructions, (
            Instruction("heading", 90, "right", "degrees"),
            Instruction("altitude", 5000, "descend", "feet"),
        ))
        self.assertEqual(result.information, (Information("track_miles", 25),))
        self.assertEqual(result.instructions[1].target_altitude_feet, 5000)
        self.assertFalse(result.information[0].readback_required)

    def test_information_can_precede_instructions_and_can_stand_alone(self):
        result = self.parse("RYR12T you have two five track miles and climb FL110")
        self.assertEqual((result.status, result.information, result.instructions), (
            "proposed", (Information("track_miles", 25),),
            (Instruction("flight_level", 110, "climb", "flight_level"),),
        ))
        result = self.parse("RYR12T you have 25 track miles")
        self.assertEqual((result.status, result.instructions, result.information),
                         ("proposed", (), (Information("track_miles", 25),)))
        result = self.parse("RYR12T you have 25 track miles then do something")
        self.assertEqual((result.status, result.information, result.unparsed),
                         ("partial", (Information("track_miles", 25),), "then do something"))
        result = self.parse("RYR12T you have twenty five track miles")
        self.assertEqual(result.information, (Information("track_miles", 25),))

    def test_ambiguous_controller_and_unmapped_spoken_controller(self):
        text = "RYR12T london control hello climb FL110"
        result = interpret_transcript("job", text, ["RYR12T"], {}, controllers={
            "LTC_N_CTR": "London Control", "LTC_S_CTR": "London Control",
        })
        self.assertEqual((result.status, result.instructions), ("ambiguous", ()))
        result = interpret_transcript("job", text, ["RYR12T"], {}, controllers={"EGKK_APP": "Gatwick Director"})
        self.assertEqual((result.status, result.identifiers), ("proposed", Identifiers("RYR12T", None, "london control")))

    def test_reported_transcripts_resolve_flight_level_without_controller_mapping(self):
        examples = (
            " Ryanair one two tango, London Control, had a climb, flight level seven zero.",
            " Ryanair one two tango London control, hello climb flight level seven zero",
        )
        for text in examples:
            for controllers in ({}, {"LTC_N_CTR": "London Control"}):
                with self.subTest(text=text, controllers=controllers):
                    result = interpret_transcript("job", text, ["RYR12T"], {"RYR": "Ryanair"},
                                                  controllers=controllers)
                    self.assertEqual(result.status, "proposed")
                    self.assertEqual(result.transcript, text)
                    self.assertEqual(result.identifiers, Identifiers(
                        "RYR12T", "LTC_N_CTR" if controllers else None, "london control",
                    ))
                    self.assertEqual(result.instructions, (Instruction("flight_level", 70, "climb", "flight_level"),))
                    self.assertEqual(result.instructions[0].target_altitude_feet, 7000)
                    self.assertEqual(bool(result.warnings), "had a" in text)

    def test_unrecognized_filler_is_reported(self):
        for text in (
            "RYR12T London control, had a descent flight level 70",
            "RYR12T London control, had a climb to altitude 7000ft",
            "RYR12T had a climb flight level 70",
        ):
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(result.status, "partial")
                self.assertTrue(result.instructions)
                self.assertEqual(result.unrecognized, ("had a",))
        result = self.parse("RYR12T London control, had a climb after departure flight level 70")
        self.assertEqual((result.status, result.instructions), ("rejected", ()))

    def test_altitude_and_ils_runway_are_explicit(self):
        result = self.parse("Ryanair 12T descend to three thousand five hundred feet and cleared ILS runway two seven right")
        self.assertEqual(result.instructions, (
            Instruction("altitude", 3500, "descend", "feet"),
            Instruction("ils", "27R"),
        ))
        result = self.parse("Ryanair 12T cleared ILS")
        self.assertEqual(result.instructions, (Instruction("ils"),))

    def test_rejects_incomplete_or_invalid_first_instruction(self):
        for text in (
            "Ryanair 12T descend to 5000",
            "Ryanair 12T turn right heading 000",
            "Ryanair 12T cleared ILS runway 37",
            "Ryanair 12T descend to 5050 feet",
            "Ryanair 12T cleared ILS runway",
        ):
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(result.status, "rejected")
                self.assertEqual(result.instructions, ())
                self.assertTrue(result.reason)

    def test_complete_prefix_is_returned_with_unparsed_remainder(self):
        examples = (
            ("Ryanair 12 tango, descend to altitude 4000 feet, QNH 1012",
             Instruction("altitude", 4000, "descend", "feet"), "qnh 1012"),
            ("Ryanair 12T turn left heading 090 and contact tower",
             Instruction("heading", 90, "left", "degrees"), "contact tower"),
            ("Ryanair 12T turn left heading 090 and",
             Instruction("heading", 90, "left", "degrees"), "and"),
            ("Ryanair 12T climb flight level 120 cleared ILS runway",
             Instruction("flight_level", 120, "climb", "flight_level"), "cleared ils runway"),
            ("Ryanair 12T climb FL150 after the waypoint",
             Instruction("flight_level", 150, "climb", "flight_level"), "after the waypoint"),
        )
        for text, instruction, unparsed in examples:
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual((result.status, result.callsign, result.instructions),
                                 ("partial", "RYR12T", (instruction,)))
                self.assertEqual(result.unparsed, unparsed)
                self.assertTrue(result.reason)
                self.assertEqual(result.transcript, text)

    def test_partial_proposal_preserves_all_complete_leading_instructions(self):
        result = self.parse("Ryanair 12T climb FL150, fly heading 080, QNH 1012")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.instructions, (
            Instruction("flight_level", 150, "climb", "flight_level"),
            Instruction("heading", 80, None, "degrees"),
        ))
        self.assertEqual(result.unparsed, "qnh 1012")

    def test_scored_parser_recovers_commands_after_unknown_words(self):
        result = self.parse("RYR12T climb FL150 some words turn left heading 090 and speed 220 knots")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.instructions, (
            Instruction("flight_level", 150, "climb", "flight_level"),
            Instruction("heading", 90, "left", "degrees"),
            Instruction("speed", 220, unit="knots"),
        ))
        self.assertEqual(result.unrecognized, ("some words",))

    def test_fuzzy_command_words_are_reported(self):
        result = self.parse("RYR12T climbe fliht level 120, turn left heding 090")
        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.instructions, (
            Instruction("flight_level", 120, "climb", "flight_level"),
            Instruction("heading", 90, "left", "degrees"),
        ))
        self.assertTrue(any("climbe" in warning for warning in result.warnings))
        self.assertTrue(any("heding" in warning for warning in result.warnings))

    def test_approximate_airline_matching_never_changes_callsign_digits(self):
        result = self.parse("Ryannair one two tango climb flight level 120")
        self.assertEqual((result.status, result.callsign), ("proposed", "RYR12T"))
        self.assertIn("airline name matched approximately", result.warnings)
        result = self.parse("Ryannair one one tango climb flight level 120")
        self.assertEqual((result.status, result.instructions), ("rejected", ()))
        result = interpret_transcript("job", "Ryannair 12T climb FL120",
                                      (cs for cs in ["RYR12T", "ABC12T"]),
                                      {"RYR": "Ryanair", "ABC": "Ryanair"})
        self.assertEqual((result.status, result.instructions), ("ambiguous", ()))

    def test_unsupported_qualifiers_are_visible_and_later_commands_are_found(self):
        result = self.parse("RYR12T climb flight level 150 after waypoint turn right heading 090")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.unrecognized, ("after waypoint",))
        self.assertEqual(len(result.instructions), 2)

    def test_invalid_speed_dme_is_not_reduced_to_unrestricted_speed(self):
        result = self.parse("RYR12T speed 160kts or less until 0dme, turn right heading 090")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.instructions, (Instruction("heading", 90, "right", "degrees"),))
        self.assertTrue(result.unrecognized)

    def test_transcript_length_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "128 tokens"):
            self.parse("RYR12T " + "word " * 129)

    def test_altitude_ft_is_recognized(self):
        result = self.parse("Ryanair 12T climb to altitude 4000ft")
        self.assertEqual((result.status, result.instructions, result.unparsed),
                         ("proposed", (Instruction("altitude", 4000, "climb", "feet"),), None))

    def test_partial_proposals_do_not_resolve_ambiguous_callsigns(self):
        result = self.parse("Ryanair 12T climb flight level 150 and contact tower",
                            ("RYR12T", "ABC12T"), {"RYR": "Ryanair", "ABC": "Ryanair"})
        self.assertEqual((result.status, result.callsign, result.instructions), ("ambiguous", None, ()))

    def test_only_exact_supplied_aircraft_and_unique_aliases_are_accepted(self):
        result = self.parse("Ryanair 12 tango turn left heading 270", ("RYR12", "RYR12T"))
        self.assertEqual((result.status, result.callsign), ("proposed", "RYR12T"))
        result = self.parse("Ryanair 12T turn left heading 270", ("RYR12",))
        self.assertEqual(result.status, "rejected")
        result = self.parse("Ryanair 12T turn left heading 270",
                            ("RYR12T", "ABC12T"), {"RYR": "Ryanair", "ABC": "Ryanair"})
        self.assertEqual((result.status, result.instructions), ("ambiguous", ()))

    def test_model_transcript_is_exported_without_mutating_the_aircraft(self):
        class Worker:
            def transcribe(self, utterance):
                self.utterance = utterance
                return "Ryanair 12T speed 180"

        worker = Worker()
        job = Utterance("utterance-42", b"\x01\x00", "", 0)
        aircraft = ["RYR12T"]
        result = transcribe_instruction(worker, job, aircraft, {"RYR": "Ryanair"})
        self.assertIs(worker.utterance, job)
        self.assertEqual(result.utterance_id, "utterance-42")
        self.assertEqual(result.instructions, (Instruction("speed", 180, unit="knots"),))
        self.assertEqual(aircraft, ["RYR12T"])

    def test_context_is_snapshotted_before_inference(self):
        aircraft = ["RYR12T"]
        airlines = {"RYR": "Ryanair"}

        class Worker:
            def transcribe(self, utterance):
                aircraft.clear()
                airlines.clear()
                return "Ryanair 12T speed 180"

        result = transcribe_instruction(Worker(), Utterance("job", b"\x01\x00", "", 0), aircraft, airlines)
        self.assertEqual((result.status, result.callsign), ("proposed", "RYR12T"))

    def test_speed_direction_is_preserved(self):
        result = self.parse("Ryanair 12T reduce speed to 180 knots")
        self.assertEqual(result.instructions, (Instruction("speed", 180, "reduce", "knots"),))

    def test_inclusive_speed_limits_and_default_knots(self):
        examples = (
            ("Ryanair 12T speed 220 kts or less", 220, "or_less"),
            ("Ryanair 12T speed 150 or greater", 150, "or_greater"),
            ("Ryanair 12T speed two one zero knots", 210, None),
        )
        for text, value, comparison in examples:
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual((result.status, result.instructions),
                                 ("proposed", (Instruction("speed", value, unit="knots", comparison=comparison),)))

    def test_explicit_icao_callsigns_do_not_require_airline_mapping(self):
        examples = ("EZY123 speed 220 kts or less", "EZY 123 speed 220 kts or less")
        for text in examples:
            with self.subTest(text=text):
                result = interpret_transcript("job", text, ["EZY123"], {})
                self.assertEqual((result.status, result.callsign, result.instructions), (
                    "proposed", "EZY123",
                    (Instruction("speed", 220, unit="knots", comparison="or_less"),),
                ))
        result = interpret_transcript("job", "EZY124 speed 220", ["EZY123"], {})
        self.assertEqual((result.status, result.callsign), ("rejected", None))

    def test_mach_values_remain_exact_and_keep_inclusive_limits(self):
        examples = (
            ("BAW 223 mach .78 or greater", "0.78", "or_greater"),
            ("BAW 223 mach point seven eight or less", "0.78", "or_less"),
            ("BAW 223 speed mach decimal 8 0", "0.80", None),
            ("BAW 223 mach 0.78", "0.78", None),
        )
        for text, value, comparison in examples:
            with self.subTest(text=text):
                result = interpret_transcript("job", text, ["BAW223"], {"BAW": "BAW"})
                self.assertEqual((result.status, result.instructions),
                                 ("proposed", (Instruction("speed", value, unit="mach", comparison=comparison),)))
        for text in ("BAW 223 mach 78", "BAW 223 mach .0"):
            with self.subTest(text=text):
                result = interpret_transcript("job", text, ["BAW223"], {"BAW": "BAW"})
                self.assertEqual((result.status, result.instructions), ("rejected", ()))

    def test_speed_limit_until_ils_dme(self):
        text = "Ryanair 12T speed 160kts or less until 4dme"
        result = self.parse(text)
        self.assertEqual((result.status, result.instructions), (
            "proposed",
            (Instruction("speed", 160, unit="knots", comparison="or_less", until=DmeTermination(4)),),
        ))
        result = self.parse("Ryanair 12T mach .78 or greater until four dme")
        self.assertEqual(result.instructions, (
            Instruction("speed", "0.78", unit="mach", comparison="or_greater", until=DmeTermination(4)),
        ))
        result = self.parse("Ryanair 12T speed 160kts or less until 0dme")
        self.assertEqual((result.status, result.instructions), ("rejected", ()))

    def test_descent_transcription_is_a_descend_instruction(self):
        result = self.parse("Ryanair 12T, descent flight level 70")
        self.assertEqual((result.status, result.instructions), (
            "proposed", (Instruction("flight_level", 70, "descend", "flight_level"),),
        ))

    def test_whisper_digit_hyphens_and_punctuation(self):
        text = " Ryanair 12 tango climb, flight level 7-0"
        result = self.parse(text)
        self.assertEqual((result.transcript, result.callsign, result.status), (text, "RYR12T", "proposed"))
        self.assertEqual(result.instructions, (Instruction("flight_level", 70, "climb", "flight_level"),))
        for value in ("70-80", "7-80", "7--0"):
            with self.subTest(value=value):
                result = self.parse(f"Ryanair 12T climb flight level {value}")
                self.assertEqual((result.status, result.instructions), ("rejected", ()))

    def test_comma_after_now_does_not_split_flight_level_instruction(self):
        text = " Ryanair 12T, London control, hello, fly heading 090 degrees, climb now, flight level 110."
        result = self.parse(text)
        self.assertEqual((result.transcript, result.status, result.unrecognized), (text, "proposed", ()))
        self.assertEqual(result.identifiers, Identifiers("RYR12T", None, "london control"))
        self.assertEqual(result.instructions, (
            Instruction("heading", 90, None, "degrees"),
            Instruction("flight_level", 110, "climb", "flight_level", timing="now"),
        ))
        for phrase in ("descend now, FL120", "climb, FL 110"):
            with self.subTest(phrase=phrase):
                result = self.parse(f"RYR12T {phrase}")
                self.assertEqual(result.status, "proposed")
                self.assertEqual(len(result.instructions), 1)
                self.assertEqual(result.instructions[0].action, "flight_level")
        result = self.parse("RYR12T climb now, heading 090")
        self.assertEqual((result.status, result.instructions, result.unrecognized), (
            "partial", (Instruction("heading", 90, None, "degrees"),), ("climb now",),
        ))

    def test_short_heading_and_flight_level_forms(self):
        examples = (
            (" Ryanair 12T, left heading 330 degrees", Instruction("heading", 330, "left", "degrees")),
            ("Ryanair 12T turn left to heading 330", Instruction("heading", 330, "left", "degrees")),
            ("Ryanair 12T turn right to 180 degrees", Instruction("heading", 180, "right", "degrees")),
            ("Ryanair 12T heading 090", Instruction("heading", 90, None, "degrees")),
            ("Ryanair 12T maintain heading zero niner zero", Instruction("heading", 90, None, "degrees")),
            ("Ryanair 12T set heading 090 degrees", Instruction("heading", 90, None, "degrees")),
            ("Ryanair 12T turn to heading 090", Instruction("heading", 90, None, "degrees")),
            (" Ryanair 12T climb FL150", Instruction("flight_level", 150, "climb", "flight_level")),
            ("Ryanair 12T descend FL 120", Instruction("flight_level", 120, "descend", "flight_level")),
        )
        for text, instruction in examples:
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual((result.status, result.instructions), ("proposed", (instruction,)))
        for text in ("Ryanair 12T left 330", "Ryanair 12T climb FL1500"):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text).status, "rejected")

    def test_altitude_requires_feet_and_keeps_flight_level_distinct(self):
        examples = (
            ("descend 4000ft", Instruction("altitude", 4000, "descend", "feet")),
            ("descend to altitude 4000ft", Instruction("altitude", 4000, "descend", "feet")),
            ("climb four thousand feet", Instruction("altitude", 4000, "climb", "feet")),
            ("descend flight level 40", Instruction("flight_level", 40, "descend", "flight_level")),
        )
        for phrase, instruction in examples:
            with self.subTest(phrase=phrase):
                result = self.parse(f"RYR12T {phrase}")
                self.assertEqual((result.status, result.instructions), ("proposed", (instruction,)))
        for phrase in ("descend 4000", "descend to altitude 4000", "heading", "heading 000"):
            with self.subTest(phrase=phrase):
                result = self.parse(f"RYR12T {phrase}")
                self.assertEqual((result.status, result.instructions), ("rejected", ()))

    def test_short_heading_preserves_unrecognized_qualifiers(self):
        result = self.parse("RYR12T after the waypoint heading 090")
        self.assertEqual((result.status, result.instructions, result.unrecognized), (
            "partial", (Instruction("heading", 90, None, "degrees"),), ("after the waypoint",),
        ))

    def test_unrecognized_greeting_does_not_get_guessed(self):
        result = interpret_transcript(
            "job", "Ryanair 12 tango London control, had I climb flight level 7-0",
            ["RYR12T"], {"RYR": "Ryanair"}, greetings=["London control hello"],
        )
        self.assertEqual((result.status, result.callsign, result.unrecognized), ("partial", "RYR12T", ("had i",)))
        self.assertEqual(result.instructions, (Instruction("flight_level", 70, "climb", "flight_level"),))

    def test_known_greeting_and_heading_direction_is_not_invented(self):
        text = " Easy one two three tango London control hello fly heading 080 degrees climb flight level 110"
        args = ("job", text, ["EZY123T"], {"EZY": "Easy"})
        self.assertEqual(interpret_transcript(*args).status, "proposed")
        result = interpret_transcript(*args, greetings=["London control hello"])
        self.assertEqual((result.status, result.callsign), ("proposed", "EZY123T"))
        self.assertEqual(result.instructions, (
            Instruction("heading", 80, None, "degrees"),
            Instruction("flight_level", 110, "climb", "flight_level"),
        ))
        unknown = interpret_transcript("job", text.replace("hello", "contact tower"), ["EZY123T"],
                                       {"EZY": "Easy"}, greetings=["London control hello"])
        self.assertEqual(unknown.status, "partial")
        self.assertEqual(unknown.unrecognized, ("contact tower",))
        self.assertEqual(len(unknown.instructions), 2)

    def test_rejection_reports_only_a_uniquely_matched_callsign(self):
        result = self.parse("Ryanair 12T contact tower")
        self.assertEqual((result.status, result.callsign), ("rejected", "RYR12T"))
        self.assertEqual(result.unrecognized, ("contact tower",))
        result = self.parse("London hello climb flight level 150")
        self.assertEqual((result.status, result.callsign), ("rejected", None))
        result = self.parse("Ryanair 12T contact tower", ("RYR12T", "ABC12T"),
                            {"RYR": "Ryanair", "ABC": "Ryanair"})
        self.assertEqual((result.status, result.callsign), ("rejected", None))

    def test_invalid_aircraft_context_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "unsupported aircraft"):
            self.parse("anything", ("not-a-callsign",))


@unittest.skipUnless(
    all(os.environ.get(name) for name in ("STT_TEST_WORKER", "STT_TEST_MODEL", "STT_TEST_WAV")),
    "set STT_TEST_WORKER, STT_TEST_MODEL and STT_TEST_WAV for the real model test",
)
class RealModelTests(unittest.TestCase):
    def test_real_model_returns_a_rejected_non_atc_transcript(self):
        worker = WhisperWorker(os.environ["STT_TEST_WORKER"], os.environ["STT_TEST_MODEL"], backend="auto")
        try:
            job = load_wav(os.environ["STT_TEST_WAV"], "")
            result = transcribe_instruction(worker, job, ["RYR12T"], {"RYR": "Ryanair"})
            self.assertEqual(result.utterance_id, job.id)
            self.assertIn("your country", result.transcript.lower())
            self.assertEqual((result.status, result.instructions), ("rejected", ()))
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
