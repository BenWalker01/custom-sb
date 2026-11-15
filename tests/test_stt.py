import io
import os
import struct
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from speech.stt import Capture, SAMPLE_RATE, WhisperWorker, load_wav, read_exact, run_microphone, save_wav


class CaptureTests(unittest.TestCase):
    def test_preroll_and_back_to_back_sessions_are_independent(self):
        capture = Capture(preroll_ms=1)
        capture.feed(b"\x01\x00" * 20)
        self.assertTrue(capture.start("first"))
        self.assertFalse(capture.start("ignored key repeat"))
        capture.feed(b"\x02\x00" * 3)
        first = capture.stop()
        capture.feed(b"\x03\x00" * 2)
        self.assertTrue(capture.start("second"))
        capture.feed(b"\x04\x00")
        second = capture.stop()
        self.assertEqual(first.pcm, b"\x01\x00" * 16 + b"\x02\x00" * 3)
        self.assertEqual(second.pcm, b"\x01\x00" * 11 + b"\x02\x00" * 3 + b"\x03\x00" * 2 + b"\x04\x00")
        self.assertEqual((first.prompt, second.prompt), ("first", "second"))
        self.assertNotEqual(first.id, second.id)
        self.assertIsNone(capture.stop())

    def test_overflow_and_device_error_discard_session(self):
        capture = Capture(preroll_ms=0, max_seconds=1)
        capture.start("")
        capture.feed(b"\x01\x00" * (SAMPLE_RATE + 1))
        with self.assertRaisesRegex(RuntimeError, "maximum duration"):
            capture.stop()
        capture.start("")
        capture.feed(b"\x01\x00", error="overflow")
        self.assertIn("overflow", capture.take_device_error())
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            capture.stop()
        capture.start("")
        capture.feed(b"\x01\x00")
        self.assertEqual(capture.stop().pcm, b"\x01\x00")

    def test_empty_and_silent_audio_are_rejected(self):
        capture = Capture(preroll_ms=0)
        capture.start("")
        with self.assertRaisesRegex(RuntimeError, "no microphone samples"):
            capture.stop()
        capture.start("")
        capture.feed(b"\0\0" * 8)
        with self.assertRaisesRegex(RuntimeError, "silent"):
            capture.stop()

    def test_preroll_alone_is_not_an_utterance(self):
        capture = Capture(preroll_ms=1)
        capture.feed(b"\x01\x00" * 16)
        capture.start("")
        with self.assertRaisesRegex(RuntimeError, "no microphone samples"):
            capture.stop()

    def test_maximum_applies_to_held_audio_even_with_empty_preroll(self):
        capture = Capture(preroll_ms=250, max_seconds=1)
        capture.start("")
        capture.feed(b"\x01\x00" * (SAMPLE_RATE + 1))
        with self.assertRaisesRegex(RuntimeError, "maximum duration"):
            capture.stop()


class WavTests(unittest.TestCase):
    def write_wav(self, path, rate, samples):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples)

    def test_load_wav_and_reject_wrong_rate_and_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            samples = struct.pack("<hh", -32768, 32767)
            self.write_wav(path, 16000, samples)
            utterance = load_wav(str(path), "hint")
            self.assertEqual(utterance.pcm, samples)
            saved = save_wav(directory, utterance)
            self.assertEqual(load_wav(str(saved), "").pcm, samples)
            with self.assertRaises(FileExistsError):
                save_wav(directory, utterance)
            self.write_wav(path, 44100, samples)
            with self.assertRaisesRegex(ValueError, "16000"):
                load_wav(str(path), "")
            self.write_wav(path, 16000, b"\0\0")
            with self.assertRaisesRegex(ValueError, "silent"):
                load_wav(str(path), "")

    def test_incomplete_worker_reply_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            read_exact(io.BytesIO(b"short"), 8)


class MicrophoneTests(unittest.TestCase):
    def test_ptt_captures_preroll_and_decodes_on_release(self):
        captured = []
        key = object()

        class FakeStream:
            active = True

            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                FakeListener.stream = self

            def __enter__(self):
                self.callback(b"\x01\x00" * 16, 16, None, None)
                return self

            def __exit__(self, *args):
                pass

        class FakeListener:
            def __init__(self, on_press, on_release):
                self.on_press = on_press
                self.on_release = on_release
                self.running = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_alive(self):
                if self.running:
                    self.running = False
                    self.on_press(key)
                    self.stream.callback(b"\x02\x00" * 2, 2, None, None)
                    self.on_release(key)
                    return True
                return False

        class FakeWorker:
            def transcribe(self, job):
                captured.append(job)
                return "example"

        sd = types.SimpleNamespace(RawInputStream=FakeStream, PortAudioError=type("PortAudioError", (Exception,), {}))
        pynput = types.SimpleNamespace(keyboard=types.SimpleNamespace(Key=types.SimpleNamespace(f8=key), Listener=FakeListener))
        args = types.SimpleNamespace(key="f8", preroll_ms=1, max_seconds=1, prompt="hint", device=None, save_wav_dir=None)
        with patch.dict("sys.modules", {"sounddevice": sd, "pynput": pynput}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "listener stopped"):
                run_microphone(FakeWorker(), args)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].pcm, b"\x01\x00" * 16 + b"\x02\x00" * 2)
        self.assertEqual(captured[0].prompt, "hint")


@unittest.skipUnless(
    all(os.environ.get(name) for name in ("STT_TEST_WORKER", "STT_TEST_MODEL", "STT_TEST_WAV")),
    "set STT_TEST_WORKER, STT_TEST_MODEL and STT_TEST_WAV for the real model test",
)
class WhisperIntegrationTests(unittest.TestCase):
    def test_worker_reuses_model_for_multiple_requests(self):
        worker = WhisperWorker(os.environ["STT_TEST_WORKER"], os.environ["STT_TEST_MODEL"])
        try:
            first = worker.transcribe(load_wav(os.environ["STT_TEST_WAV"], ""))
            second = worker.transcribe(load_wav(os.environ["STT_TEST_WAV"], "flight level, turn left"))
            self.assertIn("your country", first.lower())
            self.assertIn("your country", second.lower())
            self.assertIsNone(worker.process.poll())
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
