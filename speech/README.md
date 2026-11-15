# Standalone STT experiment

The standalone CLI displays raw transcripts. A separate Python API can propose
structured instructions from a transcript, but does not connect to `client.py`,
send network traffic, change aircraft, suppress tags, or generate readbacks.
Do not use it for real aviation.

## Python instruction proposals

`speech.instructions.transcribe_instruction` accepts an existing `WhisperWorker`
and an `Utterance`, plus aircraft callsigns and an ICAO-prefix-to-spoken-name
mapping supplied by the caller:

```python
from speech.instructions import transcribe_instruction
from speech.stt import WhisperWorker, load_wav

worker = WhisperWorker("path/to/stt-worker", "path/to/model.bin", backend="auto")
try:
    job = load_wav("path/to/recording.wav", "")
    result = transcribe_instruction(worker, job, ["RYR12T"], {"RYR": "Ryanair"})
    print(result)
finally:
    worker.close()
```

`interpret_transcript(job.id, text, aircraft, airlines)` offers the same
conversion when a transcript is already available. For microphone users,
`run_microphone(worker, args, on_transcript=callback)` invokes
`callback(job, text)` on the serial decoder thread; the callback can call
`interpret_transcript`. It should hand off results quickly and must not mutate
simulator state. A callback error stops decoding and is reported.
`transcribe_instruction` copies the supplied aircraft and airline mappings
before inference; callers needing PTT-press context must capture that snapshot
themselves before submitting the utterance.

An optional `greetings=["London control hello"]` argument to either API permits
that exact greeting between a callsign and an instruction. Greetings are not
ignored by default, and unconfigured wording is rejected. Aircraft and spoken
airline names must still be supplied explicitly; for example, matching “Easy
one two three tango” requires an aircraft such as `EZY123T` and a matching
spoken name such as `{"EZY": "Easy"}`. This mapping is only an example of a
transcript alias, not proof of what was spoken.

The immutable `TranscriptResult` contains `utterance_id`, the unmodified
`transcript`, `status`, `callsign`, `instructions`, `reason`, and `unparsed`.
Status is `proposed`, `partial`, `rejected`, or `ambiguous`; rejections and
ambiguities contain no instructions. A `partial` result contains the complete
recognized instructions before the first unsupported or invalid part, plus
the normalized unparsed remainder and an explanatory reason. It does not skip
over unknown words to find later commands. Each `Instruction` has an `action`,
`value`, `direction`, `unit`, `comparison`, and `until`. Speed `comparison`
is `at_most` for “or less”, `at_least` for “or greater”, or `None` for an
exact speed; these limits are inclusive. Unspecified speed units mean knots.
Mach values are decimal strings, such as `"0.78"`, to preserve their value.
“Until 4 DME” sets `until` to `DmeTermination(distance_nm=4, reference="ils")`;
it does not identify a particular runway or ILS facility. An ILS proposal with
no spoken runway has `value=None`; it does not
identify or authorize a runway. A “fly heading” instruction without a stated
turn direction has `direction=None`; the API does not choose a turn direction.
If the callsign is ambiguous or the first instruction cannot be understood,
the result contains no instructions. A partial result can omit words that
would have changed the meaning of its recognized prefix; it is not proof of
the full intended instruction.

The current deliberately restricted grammar accepts an exact supplied
callsign followed by flight-level climb/descend (including `FL150`), altitude
climb/descend in feet, left/right heading (including “left heading”), speed,
and ILS clearance. For example: “Ryanair one
two tango, turn right heading zero niner zero, descend flight level one two
zero and speed 210 knots.” Speeds accept knots, `kt`, `kts`, and Mach values
such as “Mach point seven eight” or “Mach .78 or greater”, optionally followed
by “until 4 DME”. Altitudes accept `ft` as well as `feet`. Digits can
be a written number or separately spoken
digits; single digits separated by hyphens, such as `7-0`, are also treated as
a digit sequence. Altitude also accepts forms such as “three thousand five hundred
feet”. Runway sides accept left, right, and centre. Other wording, including
number words such as “twenty”, may be rejected. Callsigns must have a
three-letter ICAO prefix, digits, and optional letter suffix. Exact ICAO
callsigns such as `EZY123` and `BAW 223` need no spoken airline mapping.
Airline-name callsigns require an explicit mapping. Hints passed to the CLI
are not an aircraft mapping.

These are **proposals**, not verified clearances. Recognition can hallucinate
or mishear even an apparently valid instruction. A simulator must independently
resolve and revalidate the aircraft, authorization, proposed commands, values,
runway and live state before applying anything; it must report any unparsed
remainder and exactly what was applied. The simulator currently does not
enforce Mach bounds or ILS DME restrictions; it must implement and validate
those semantics rather than treating a bound as an exact speed. This API does not
implement that integration or provide an aircraft context snapshot.

## Build

Use a local whisper.cpp checkout and a compatible ggml model. Check model
provenance and licensing before downloading or distributing weights. The
adjacent checkout mentioned in the plan already has a stock English model; the
ATC-trained weights mentioned in `../whisper-atc.md` require separate license
verification. From the repository root, for example:

```sh
cmake -S speech -B speech-build -DWHISPER_CPP_DIR="$PWD/../whisper.cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build speech-build --target stt-worker --config Release
```

On Windows, use an absolute path for `WHISPER_CPP_DIR` and the executable
produced by the build (often `speech-build/bin/Release/stt-worker.exe`). A GPU
backend must be enabled in the whisper.cpp build to use it; CPU is the default.
Build outside the repository or remove your own build directory when finished.

## Try a recording

```sh
python speech/stt.py --worker speech-build/bin/stt-worker \
  --model ../whisper.cpp/models/ggml-base.en.bin \
  --wav ../whisper.cpp/samples/jfk.wav \
  --prompt "flight level, turn left, turn right"
```

Input WAVs must be mono, 16 kHz, uncompressed PCM16, and at most 30 seconds.
The model loads and warms up once. The sample is useful as a smoke test, not
an ATC accuracy benchmark. `--prompt` is an optional static hint; no simulator
state is connected. Compare with and without hints, including misleading ones.
Try `--beam 3` for explicit beam search, `--no-fallback` to disable temperature
fallback, or `--backend auto` for the backend provided by your build.

For callsigns, provide the spoken airline name and flight number as a hint:

```sh
python speech/stt.py --worker speech-build-cuda/bin/stt-worker \
  --model ../whisper.cpp/models/ggml-medium.en-q5_0.bin \
  --backend auto --no-fallback --callsign "Ryanair 12T"
```

Repeat `--callsign` for other aircraft. The prototype generates both numeric
and phonetic variants such as "Ryanair 12 tango" and "Ryanair one two tango".
Supply telephony names, not airline codes such as `RYR`; use `--prompt` for
other names or unusual callsign formats. Hints bias Whisper but do not
guarantee accuracy and can make it hear a callsign that was not spoken.
Compare the same recording with and without hints, including misleading
hints; there is no simulator-derived aircraft list yet. Long prompts may be
truncated by Whisper, so keep the list short.

## Faster inference with a GPU

The normal build and default `--backend cpu` use the CPU. To use an NVIDIA GPU,
install a compatible CUDA toolkit and build a separate CUDA-enabled worker:

```sh
cmake -S speech -B speech-build-cuda \
  -DWHISPER_CPP_DIR="$PWD/../whisper.cpp" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build speech-build-cuda --target stt-worker --config Release -j 4
python speech/stt.py --worker speech-build-cuda/bin/stt-worker \
  --model ../whisper.cpp/models/ggml-medium.en-q5_0.bin \
  --backend auto --no-fallback
```

The model must already be present. `--backend auto` requests the GPU but
whisper.cpp can fall back to CPU; check startup output for a CUDA device and
`using CUDA0 backend`. A CPU-only build cannot use the GPU even with this
option. The Windows executable is usually under `bin/Release`.
VICE disables temperature fallback to avoid repeated decoding; `--no-fallback`
does the same, but may reduce recognition accuracy. Omit it to compare.
Both methods keep the model loaded, warm it up and decode each complete PTT
utterance once. Actual latency depends on the model, hardware and audio;
compare the displayed inference and total timings on the same recordings.

## Try push-to-talk

Install the optional microphone dependencies in your Python environment:

```sh
python -m pip install -r speech/requirements.txt
python speech/stt.py --worker speech-build/bin/stt-worker \
  --model ../whisper.cpp/models/ggml-base.en.bin --key f8
```

Hold F8, speak, and release to transcribe. Use `--key` for another pynput
special key; a graphical session/keyboard permissions may be needed for global
hotkeys. Select a microphone with `--device` (name or numeric PortAudio index;
list devices with `python -m sounddevice`). The device must support mono
16 kHz PCM16 capture; native-rate resampling is not implemented. Default
pre-roll is 250 ms and maximum held audio is 15 s (`--preroll-ms` and
`--max-seconds`). If capture overruns or exceeds the limit, the entire
utterance is rejected. Two pending utterances can queue while one decodes;
additional submissions are visibly rejected.

Audio stays in memory and is sent only to the local worker process; transcripts
are printed to the terminal. No recordings are saved unless explicitly enabled
with `--save-wav-dir PATH`, where PATH is an existing directory. This writes
one WAV per queued utterance, named by utterance ID, for inspecting capture;
the user is responsible for deleting these recordings. Exact digital silence is
rejected, but noisy silence may still produce hallucinated text. Avoid speaker
playback feeding back into the microphone. This prototype does not implement
TTS/playback arbitration, model download, automatic fallback, or accuracy
validation. It captures in 20 ms blocks with no release tail, so the final
block may be clipped around release; check saved audio before integration.
Measure release-to-result latency and recognition accuracy with consented
representative ATC recordings before integration.

Run isolated tests with `python -m unittest discover -s tests -p test_stt.py`.
To include the real-model integration test, set `STT_TEST_WORKER`,
`STT_TEST_MODEL`, and `STT_TEST_WAV` to the built worker, model, and
whisper.cpp's `samples/jfk.wav`, respectively.
