"""Standalone PTT transcription experiment; no simulator integration."""

import argparse
from array import array
from dataclasses import dataclass
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import uuid
import wave

SAMPLE_RATE = 16000
MAX_SAMPLES = 30 * SAMPLE_RATE
MAX_PROMPT_BYTES = 16384


@dataclass(frozen=True)
class Utterance:
    id: str
    pcm: bytes
    prompt: str
    released_at: float


class Capture:
    """Thread-safe PCM16 capture with a pre-roll and independent PTT sessions."""

    def __init__(self, preroll_ms=250, max_seconds=15):
        if not 0 <= preroll_ms <= 1000 or not 1 <= max_seconds <= 29:
            raise ValueError("pre-roll must be 0-1000 ms and maximum duration 1-29 s")
        self.lock = threading.Lock()
        self.preroll_bytes = SAMPLE_RATE * preroll_ms // 1000 * 2
        self.max_bytes = SAMPLE_RATE * max_seconds * 2
        self.recent = bytearray()
        self.parts = None
        self.length = 0
        self.invalid = None
        self.prompt = ""
        self.device_error = None

    def feed(self, pcm, error=None):
        with self.lock:
            if error:
                self.device_error = f"microphone input error: {error}"
                self.recent.clear()
                if self.parts is not None:
                    self.invalid = self.device_error
                return
            if self.parts is not None and not self.invalid:
                if self.length + len(pcm) > self.max_bytes:
                    self.invalid = "utterance exceeded maximum duration"
                else:
                    self.parts.append(bytes(pcm))
                    self.length += len(pcm)
            self.recent.extend(pcm)
            if len(self.recent) > self.preroll_bytes:
                del self.recent[:len(self.recent) - self.preroll_bytes]

    def take_device_error(self):
        with self.lock:
            error, self.device_error = self.device_error, None
            return error

    def start(self, prompt):
        with self.lock:
            if self.parts is not None:
                return False
            self.parts = [bytes(self.recent)]
            self.length = 0
            self.invalid = None
            self.prompt = prompt
            return True

    def stop(self):
        with self.lock:
            if self.parts is None:
                return None
            parts, error, prompt, length = self.parts, self.invalid, self.prompt, self.length
            self.parts = None
            released_at = time.monotonic()
        if error:
            raise RuntimeError(error)
        if not length:
            raise RuntimeError("no microphone samples captured while PTT was held")
        pcm = b"".join(parts)
        if not pcm or not any(pcm):
            raise RuntimeError("empty or digitally silent recording")
        return Utterance(uuid.uuid4().hex, pcm, prompt, released_at)


def read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise RuntimeError("Whisper worker exited or returned an incomplete response")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class WhisperWorker:
    def __init__(self, executable, model, backend="cpu", beam=1, fallback=True):
        self.process = subprocess.Popen(
            [executable, model, backend, str(beam), "fallback" if fallback else "no-fallback"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        try:
            if self._reply() != "ready":
                raise RuntimeError("unexpected Whisper worker startup response")
        except Exception:
            self.close()
            raise

    def _reply(self):
        status, size = struct.unpack("<II", read_exact(self.process.stdout, 8))
        if size > 1024 * 1024:
            raise RuntimeError("Whisper worker response is too large")
        text = read_exact(self.process.stdout, size).decode("utf-8")
        if status:
            raise RuntimeError(text)
        return text

    def transcribe(self, utterance):
        prompt = utterance.prompt.encode("utf-8")
        if not utterance.pcm or len(utterance.pcm) % 2 or len(utterance.pcm) // 2 > MAX_SAMPLES:
            raise ValueError("expected 1-480000 mono PCM16 samples")
        if len(prompt) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds 16384 UTF-8 bytes")
        self.process.stdin.write(struct.pack("<II", len(utterance.pcm) // 2, len(prompt)))
        self.process.stdin.write(prompt)
        self.process.stdin.write(utterance.pcm)
        self.process.stdin.flush()
        return self._reply()

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


def load_wav(path, prompt):
    with wave.open(path, "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("WAV must be uncompressed mono 16-bit PCM at 16000 Hz")
        if not 0 < wav.getnframes() <= MAX_SAMPLES:
            raise ValueError("WAV must contain between 1 sample and 30 seconds")
        expected = wav.getnframes() * 2
        pcm = wav.readframes(wav.getnframes())
        if len(pcm) != expected:
            raise ValueError("WAV contains incomplete audio")
    if not any(pcm):
        raise ValueError("WAV is digitally silent")
    return Utterance(uuid.uuid4().hex, pcm, prompt, time.monotonic())


def save_wav(directory, utterance):
    path = Path(directory) / f"{utterance.id}.wav"
    with path.open("xb") as output:
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(utterance.pcm)
    return path


def run_microphone(worker, args):
    try:
        import sounddevice as sd
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError(f"Microphone/keyboard dependency unavailable: {exc}") from exc

    key = getattr(keyboard.Key, args.key, None)
    if key is None:
        raise ValueError(f"Unknown pynput special key: {args.key}")
    capture = Capture(args.preroll_ms, args.max_seconds)
    jobs = queue.Queue(maxsize=2)
    results = queue.Queue()

    def decode():
        while True:
            job = jobs.get()
            if job is None:
                return
            queue_wait = time.monotonic() - job.released_at
            if args.save_wav_dir:
                try:
                    path = save_wav(args.save_wav_dir, job)
                    results.put(f"[{job.id}] Saved audio: {path}")
                except OSError as exc:
                    results.put(f"[{job.id}] ERROR saving audio: {exc}")
            started = time.monotonic()
            try:
                text = worker.transcribe(job)
                results.put(f"[{job.id}] {text.strip()} (queue {queue_wait:.2f}s, inference {time.monotonic() - started:.2f}s, total {time.monotonic() - job.released_at:.2f}s)")
            except (OSError, RuntimeError, ValueError) as exc:
                results.put(f"[{job.id}] ERROR: {exc}")

    def callback(indata, frames, timing, status):
        if sys.byteorder != "little":
            samples = array("h")
            samples.frombytes(indata)
            samples.byteswap()
            indata = samples.tobytes()
        capture.feed(indata, str(status) if status else None)

    def on_press(pressed):
        if pressed == key and capture.start(args.prompt):
            results.put("Recording...")

    def on_release(released):
        if released == key:
            try:
                job = capture.stop()
                if job:
                    jobs.put_nowait(job)
                    results.put(f"[{job.id}] Processing...")
            except (RuntimeError, queue.Full) as exc:
                results.put(f"ERROR: {'inference queue is full' if isinstance(exc, queue.Full) else exc}")

    thread = threading.Thread(target=decode)
    thread.start()
    try:
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                   device=args.device, blocksize=320, callback=callback) as stream:
                with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                    print(f"Hold {args.key} to record; release to transcribe. Ctrl+C to exit.", flush=True)
                    while listener.is_alive():
                        if not stream.active:
                            raise RuntimeError("Microphone stream stopped unexpectedly")
                        error = capture.take_device_error()
                        if error:
                            print(f"ERROR: {error}", flush=True)
                        try:
                            print(results.get(timeout=0.2), flush=True)
                        except queue.Empty:
                            pass
                    raise RuntimeError("Keyboard listener stopped unexpectedly")
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Microphone unavailable: {exc}") from exc
    finally:
        jobs.put(None)
        thread.join()
        while not results.empty():
            print(results.get_nowait(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, help="path to built stt-worker")
    parser.add_argument("--model", required=True, help="path to a compatible ggml Whisper model")
    parser.add_argument("--backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--beam", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--prompt", default="", help="optional static context hint, not a grammar")
    parser.add_argument("--wav", help="transcribe an existing mono 16 kHz PCM16 WAV instead of microphone")
    parser.add_argument("--key", default="f8", help="pynput special key for PTT (default: f8)")
    parser.add_argument("--device", help="PortAudio input device name or numeric index")
    parser.add_argument("--preroll-ms", type=int, default=250)
    parser.add_argument("--max-seconds", type=int, default=15)
    parser.add_argument("--save-wav-dir", help="opt in to saving captured audio in an existing directory")
    args = parser.parse_args(argv)
    if args.device and args.device.isdecimal():
        args.device = int(args.device)
    try:
        if len(args.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds 16384 UTF-8 bytes")
        if args.save_wav_dir and not Path(args.save_wav_dir).is_dir():
            raise ValueError("audio output directory must already exist")
        if args.wav and args.save_wav_dir:
            raise ValueError("--save-wav-dir is only available for microphone capture")
        job = load_wav(args.wav, args.prompt) if args.wav else None
        print("Loading and warming up local Whisper model...", file=sys.stderr, flush=True)
        worker = WhisperWorker(args.worker, args.model, args.backend, args.beam, not args.no_fallback)
        try:
            if job:
                started = time.monotonic()
                text = worker.transcribe(job)
                print(text.strip())
                print(f"inference {time.monotonic() - started:.2f}s", file=sys.stderr)
            else:
                run_microphone(worker, args)
        finally:
            worker.close()
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, wave.Error) as exc:
        print(f"STT error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
