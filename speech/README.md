# Standalone STT experiment

This prototype only displays raw transcripts. It does not connect to
`client.py`, send network traffic, interpret instructions, change aircraft,
suppress tags, or generate readbacks. Do not use it for real aviation.

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
