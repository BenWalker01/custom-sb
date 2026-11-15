import os
import unittest

from speech.instructions import DmeTermination, Instruction, interpret_transcript, transcribe_instruction
from speech.stt import Utterance, WhisperWorker, load_wav


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
            "Ryanair 12T turn right heading 090?",
            "Ryanair 12T cleared ILS runway",
            "Ryanair 12T go turn left heading 330 degrees, descend flight level 150",
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
             Instruction("heading", 90, "left", "degrees"), "and contact tower"),
            ("Ryanair 12T turn left heading 090 and",
             Instruction("heading", 90, "left", "degrees"), "and"),
            ("Ryanair 12T descend flight level 120 climb flight level 150",
             Instruction("flight_level", 120, "descend", "flight_level"), "climb flight level 150"),
            ("Ryanair 12T climb flight level 120 cleared ILS runway",
             Instruction("flight_level", 120, "climb", "flight_level"), "cleared ils runway"),
            ("Ryanair 12T climb FL150 after the waypoint",
             Instruction("flight_level", 150, "climb", "flight_level"), "after the waypoint"),
            ("Ryanair 12T climb FL150 unknown turn left heading 090",
             Instruction("flight_level", 150, "climb", "flight_level"), "unknown turn left heading 090"),
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
            ("Ryanair 12T speed 220 kts or less", 220, "at_most"),
            ("Ryanair 12T speed 150 or greater", 150, "at_least"),
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
                    (Instruction("speed", 220, unit="knots", comparison="at_most"),),
                ))
        result = interpret_transcript("job", "EZY124 speed 220", ["EZY123"], {})
        self.assertEqual((result.status, result.callsign), ("rejected", None))

    def test_mach_values_remain_exact_and_keep_inclusive_limits(self):
        examples = (
            ("BAW 223 mach .78 or greater", "0.78", "at_least"),
            ("BAW 223 mach point seven eight or less", "0.78", "at_most"),
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
            (Instruction("speed", 160, unit="knots", comparison="at_most", until=DmeTermination(4)),),
        ))
        result = self.parse("Ryanair 12T mach .78 or greater until four dme")
        self.assertEqual(result.instructions, (
            Instruction("speed", "0.78", unit="mach", comparison="at_least", until=DmeTermination(4)),
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

    def test_short_heading_and_flight_level_forms(self):
        examples = (
            (" Ryanair 12T, left heading 330 degrees", Instruction("heading", 330, "left", "degrees")),
            ("Ryanair 12T turn left to heading 330", Instruction("heading", 330, "left", "degrees")),
            ("Ryanair 12T turn right to 180 degrees", Instruction("heading", 180, "right", "degrees")),
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

    def test_unrecognized_greeting_does_not_get_guessed(self):
        result = interpret_transcript(
            "job", "Ryanair 12 tango London control, had I climb flight level 7-0",
            ["RYR12T"], {"RYR": "Ryanair"}, greetings=["London control hello"],
        )
        self.assertEqual((result.status, result.callsign, result.instructions), ("rejected", "RYR12T", ()))

    def test_greeting_is_opt_in_and_heading_direction_is_not_invented(self):
        text = " Easy one two three tango London control hello fly heading 080 degrees climb flight level 110"
        args = ("job", text, ["EZY123T"], {"EZY": "Easy"})
        self.assertEqual(interpret_transcript(*args).status, "rejected")
        result = interpret_transcript(*args, greetings=["London control hello"])
        self.assertEqual((result.status, result.callsign), ("proposed", "EZY123T"))
        self.assertEqual(result.instructions, (
            Instruction("heading", 80, None, "degrees"),
            Instruction("flight_level", 110, "climb", "flight_level"),
        ))
        unknown = interpret_transcript("job", text.replace("hello", "contact tower"), ["EZY123T"],
                                       {"EZY": "Easy"}, greetings=["London control hello"])
        self.assertEqual((unknown.status, unknown.instructions), ("rejected", ()))

    def test_rejection_reports_only_a_uniquely_matched_callsign(self):
        result = self.parse("Ryanair 12T contact tower")
        self.assertEqual((result.status, result.callsign), ("rejected", "RYR12T"))
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
