# Speech-driven controller instructions plan

## Goal and scope

Add push-to-talk (PTT) speech-to-text (STT) for controller instructions and
text-to-speech (TTS) for simulated pilot responses. The first release is for
one human controller, connecting through `client/client.py`, even when the
client and simulator run on the same machine. Each player's client should run
its own STT and TTS locally; the host runs the simulation, not one model per
player. Design the interfaces for multiple controllers, but do not implement
multiplayer voice or shared radio in the first release.

When a controller is in speech mode, **ignore aircraft instructions originating
from that controller's Euroscope tags**, including while speech is processing or
unavailable. Never execute both the tag instruction and the spoken version.
Keep existing tag behavior for controllers in tag mode and preserve non-command
FSD traffic. Do not silently fall back to tags on a speech failure: require an
explicit mode change.

This is a simulator feature, not a system for real-world aviation.

## Existing integration points

- `client/client.py` is currently a byte-forwarding TCP proxy between one local
  Euroscope connection and a hard-coded remote FSD address. It has no speech,
  configuration, framing, controller identity, or control channel.
- `sb.py` runs the FSD server, tracks controllers and pilots, and forwards
  controller messages to other controllers. `main.py` connects as the simulated
  master controller and receives those forwarded messages.
- `main.py:messageMonitor` turns tag-originated `$CQ ... :SC:`, `:TA:`, `:BC:`,
  `:DR:`, and other direct/hold/ILS/level messages into `parseCommand` calls.
  `$AM` can also cause a STAR command. Handoff `$HO`/`$HA`, flight-plan
  messages and `#TM` administrative behavior have separate effects. Classify
  the actual message forms and sender before filtering; do not suppress every
  `$CQ` or all traffic from a speech controller.
- `main.py:parseCommand` mutates `Plane` objects directly and mostly prints
  failures; it does not return a reliable success result. Some operations
  perform multiple mutations or catch errors internally. The existing command
  descriptions are in `commandSpec.txt`; actual behavior in `main.py` is
  authoritative where they differ.
- `Plane.currentlyWithData` and `firstController` help with simulated handoffs
  but are not a complete authoritative human-controller/frequency ownership
  model. `ACTIVE_CONTROLLERS` is a static list, not a live speech-mode registry.
- `pyttsx3` is already a dependency, but the old `_TTS`/`keyboardHandler` path
  is disabled, readback collection is commented out, and shared
  `currentSpeakingAC`/`messagesToSpeak` are unsuitable for concurrent jobs.
  `voiceTesting.py` is only a PocketSphinx experiment, not an integrated STT
  pipeline.
- `main.py` runs a blocking position/message loop, with a one-second socket
  select and a sleep between updates; other timer threads also mutate state.
  This affects both latency and safe command execution.

## Proposed architecture

```text
Controller machine                          Simulator host
Euroscope -- FSD --> client.py proxy ------> sb.py --> main.py
                        |                                |
 microphone -> PTT -> local STT               authoritative sim state
                        |                                |
                        +-- final transcript + ID ------>+-> interpret,
                        |    over speech control channel |   validate, apply
 local TTS <- readback text + outcome <------------------+
```

The client owns microphone capture, the loaded Whisper model, and playback.
Send final text and metadata, not microphone audio or model data, to the host.
The host resolves the transcript using current simulation context, validates
and applies it, then sends a structured outcome and pilot readback text to the
originating client. Local TTS speaks the readback. This keeps the simulator
authoritative without requiring the host to run each player's models.

Use a **separate, versioned speech control channel**, not ad-hoc messages
injected into the Euroscope FSD byte stream. Initially it may be a TCP
connection from `client.py` to a small service hosted with `main.py`; define
explicit framing, bounded message sizes, timeouts and reconnect behavior.
Keep `sb.py`'s existing FSD forwarding compatible. Do not assume one TCP
`recv` equals one message: buffer partial frames and handle multiple frames
per read in both the new channel and any proxy changes.

Single-machine setup needs separate configurable listener ports: `sb.py`
currently binds `127.0.0.1:6809` and the proxy also binds that address. For
example, keep FSD at 6809, listen for Euroscope on another local proxy port,
and use a third port for speech control. Make local/remote host and port
settings explicit; remove reliance on the hard-coded ngrok endpoint. Verify
the end-to-end startup instructions, including how `main.py` reaches `sb.py`.

### Identity, mode and connection lifecycle

- Bind each speech session to the actual FSD controller connection/identity,
  not a callsign supplied in each transcript. Design an explicit registration
  and association mechanism between the proxy, FSD login and control channel;
  enforce it on the host. A practical first version can use a host-issued,
  single-use session token and verified FSD login, with an explicit enrollment
  step if necessary. Specify and test how that token is delivered and how
  reconnects invalidate it before implementing; do not claim a self-reported
  callsign is authentication.
- Single-player may bind the service to loopback by default. For remote use,
  require authenticated, encrypted transport or a trusted encrypted tunnel;
  do not expose an unauthenticated command endpoint to the internet. Avoid
  logging tokens or FSD credentials.
- Maintain a host-owned per-controller mode (`tag` or `speech`) and session
  generation. Only enable speech mode after the local model, audio input and
  control channel are ready and the host acknowledges the mode. Establish an
  ordered activation boundary: once activation is acknowledged, later tag
  instructions from that controller must not execute. Make mode changes
  explicit and observable.
- Filter at the point where FSD messages become simulator instructions, based
  on the verified sender and mode; continue forwarding messages to Euroscope.
  Cover every tag-to-command branch, including `$AM` STAR behavior and `#TM`
  only where it actually represents an aircraft instruction. Distinguish
  flight-plan synchronization, handoffs, administrative actions, and
  simulator-generated messages from human tag commands by inspecting their
  sender and purpose. Preserve their existing behavior.
- If the client disconnects, the model fails, or a request times out, leave
  speech mode in a visibly unavailable, fail-closed state; do not replay
  buffered tag commands or automatically switch modes. Reconnection or an
  explicit switch to tag mode creates a new generation and rejects stale
  in-flight speech results. Inform the controller of the current status.

### Speech control contract

Define a small schema with a protocol version, authenticated session,
controller identity derived by the host, session generation, unique
utterance ID, and message type. Support readiness/mode negotiation, a context
snapshot request, final transcript submission, and command outcome/readback.
An outcome should contain the utterance ID, status (`applied`, `rejected`, or
`ambiguous`), recognized text, interpreted instruction(s), explanatory reason
when needed, and readback text only for what was actually applied.

Use bounded queues, explicit overload errors, and per-session ordering so
back-to-back utterances cannot overtake one another. Deduplicate IDs on the
host within the session; a retry returns the prior outcome, never reapplies a
command. If a connection drops after submission, query the outcome by ID after
reconnect rather than assuming success or replaying blindly. Reject expired
or stale-generation requests. Specify a bounded retention/expiry policy and
show an unknown outcome rather than executing again when it cannot be proven
safe.

## Implementation steps

### 1. Establish a single authoritative command path

1. Inventory the tag packet variants and their sender fields using existing
   code and captured/test messages. Extract classification from
   `messageMonitor` so filtering is per controller and per instruction, before
   any associated mutation. Ensure each incoming packet is processed once,
   rather than accidentally once per `ACTIVE_CONTROLLERS` loop iteration.
2. Refactor command application into an explicit API with source/controller,
   typed command, and structured success or failure. Existing manual and tag
   callers should use the same validation/application path while retaining
   their current behavior in tag mode. Keep admin commands separate.
3. Validate complete arguments and aircraft state before mutation; avoid
   success-shaped responses from the current exception handlers. A compound
   transmission needs an explicit policy: validate the whole set before
   applying it, or report exactly which parts were applied. Prefer atomic
   application; reject unsupported or ambiguous parts rather than silently
   executing a subset.
4. Serialize incoming speech requests onto the simulation's state-owning
   execution path. Rework the blocking loop/wakeup and shared timer mutations
   as needed so command handling does not wait a full radar interval or race
   plane spawning, handoffs, and position updates. Do not mutate planes in
   network, audio, inference, or playback callbacks.

### 2. Add the client and host control channel

1. Make `client/client.py` configurable and keep its FSD proxy working without
   speech. Add the separate host-side speech service and the session/mode
   contract above. Integrate its lifecycle with startup and shutdown of
   `main.py`, including clean disconnects and explicit errors.
2. Create an immutable context snapshot on PTT press containing the eligible
   aircraft callsigns and relevant spoken aliases, route fixes, active runways,
   approach terminology and frequency/controller context that actually exist
   in this simulation. Limit its size; do not assume the current ownership
   fields or FSD frequency encoding are sufficient without verifying them.
   For single-player, define a conservative eligible-aircraft policy.
3. Context is only an STT hint. At execution, recheck aircraft existence,
   authorization, current mode, instruction validity and relevant state
   against the live simulator; do not execute based solely on a stale
   snapshot. A missing/stale context must be visible and must not broaden
   authority or cause guessed callsigns.

### 3. Capture and recognize speech locally

1. Choose and test a maintained Python binding or local persistent service
   compatible with the intended whisper.cpp model format, quantization,
   language, prompt and hardware backend. Confirm platform support and model
   provenance, license and redistribution rights before distribution. Provide
   a documented local model path, a CPU option and clear load/warmup errors.
   Compare an eligible ATC-trained English model against a stock model rather
   than assuming it is better.
2. Add configurable microphone selection and PTT press/release handling in
   the client, compatible with the target OS and Euroscope focus. Capture a
   bounded 200–300 ms experimental pre-roll plus the complete utterance;
   verify no gaps/duplicates at boundaries. Convert/resample to the chosen
   backend's mono 16 kHz format correctly. Bound utterance duration and buffer
   sizes; surface device loss and overflow.
3. Keep the model loaded and submit one complete utterance on release to a
   bounded, serialized worker. Store separate audio, prompt snapshot, ID and
   timing per press, even if another utterance is decoding. Keep callbacks
   lightweight and UI/proxy responsive. Show recording, processing and error
   state. Reject overload explicitly.
4. Use English, no translation or prior-utterance context, and a compact prompt
   derived from the snapshot. Verify token/truncation behavior for the chosen
   backend. Compare greedy and explicitly selected beam search, and
   temperature fallback settings, on representative recordings. Do not treat
   prompt hints or model confidence as proof of what was said.
5. Send only final transcripts for interpretation; provisional live captions
   are optional later and must never trigger execution. Ignore empty/silent
   recordings and prevent TTS playback leaking into microphone capture, using
   headphones and/or an explicit PTT/playback arbitration policy.

### 4. Interpret, validate and respond

1. Implement a separate aviation interpreter: normalize spoken callsigns,
   phonetics and numbers; resolve against eligible aircraft and route/runway
   context; produce typed candidates rather than guessing strings for
   `parseCommand`. Preserve the original transcript for diagnostics.
2. Start with a defined, testable subset: climb/descend flight level, left/right
   heading, speed, direct/resume own navigation, squawk, hold and ILS where
   existing simulator semantics support them. Explicitly reject unsupported
   altitude units, runway choices, commands and combinations. Expand to STAR,
   ground and other `commandSpec.txt`/`parseCommand` operations in later
   increments, after their existing semantics and validation are understood.
   Map flight levels to the current hundreds-of-feet representation
   deliberately; distinguish heading direction, digit sequences and squawk
   leading zeroes.
3. Never guess a missing digit, callsign, fix or action. On ambiguity or
   invalid state, return a visible rejection/request for repetition or an
   explicit confirmation flow; do not mutate a plane or play a success
   readback. Revalidate any confirmed proposal against current state.
4. Generate readback from the **applied structured result**, not the raw
   transcript or tag. Include callsign and applicable cleared values; report
   rejected instructions separately. Add a serialized, bounded local TTS
   playback queue, initially evaluating existing `pyttsx3` on supported
   platforms. Specify device/voice selection, interruption and failure
   behavior; a TTS failure must not undo an already-applied command or cause a
   retry. A text outcome remains visible.

### 5. Prepare, but do not implement, multiplayer

Keep protocol identity, session generation, mode, context, queues and outcomes
per controller from the start. In a later release, add authoritative
controller-to-aircraft/frequency ownership and handoff updates, per-frequency
visibility and readback routing, authentication, and a policy for simultaneous
transmissions, PTT arbitration and ordering. If controllers should hear one
another, that requires a separate radio-audio transport and privacy design;
the transcript/readback channel alone does not provide it. Multiple local
models run on the players' machines, not on the host.

## Verification and acceptance

- Unit-test message classification and mode transitions for each actual
  tag-to-command path. In speech mode tags from that controller cause no
  aircraft command, including after disconnect or inference failure; in tag
  mode existing commands still work. Handoffs, flight-plan synchronization,
  admin messages, simulator-generated commands, and other controllers'
  traffic remain functional. Test activation boundaries, sender matching and
  stale sessions.
- Test transcript normalization and validation with representative callsigns,
  aliases, spoken digits, headings, FLs, speeds, fixes, runways, squawks,
  compound commands, ambiguities, missing values and changed/deleted
  aircraft. Assert exact state changes or no change, and that readbacks match
  applied results.
- Integration-test `client.py` -> `sb.py` -> `main.py` plus the speech control
  channel on one machine with distinct ports. Use fake microphone/STT/TTS
  backends for deterministic tests: speech changes the intended plane exactly
  once, a simultaneous tag does not, readback reaches the right client, and
  legacy tag mode still works. Exercise split/coalesced network frames,
  retries, disconnects, duplicate IDs, queue overload, quick PTT presses,
  device loss, and TTS failure.
- With consent, collect representative recordings and manually checked
  transcripts/expected commands, including silence, clipped words, similar
  callsigns, different microphones and accents, and misleading prompt hints.
  Keep development and held-out examples separate. Measure field-level
  callsign/action/heading/altitude/speed/fix accuracy, false executions,
  misses and ambiguity, alongside p50/p95 PTT-release-to-outcome latency,
  queue wait, inference and playback delay under realistic simulator load.
  Agree project-specific accuracy and latency targets before selecting a
  model; do not equate transcript accuracy with command accuracy.
- Document installation, model licensing/download verification, audio/PTT
  setup, ports, startup, mode status, privacy and troubleshooting. Audio
  stays local; diagnostics/audio recording must be opt-in, with retention
  stated. Update `requirements.txt` and `client/client.spec` as needed, and
  verify the packaged client on the supported target platform.

## Decisions to settle before implementation

- Supported client OS/Python versions, microphone/PTT device and key behavior;
  whether a separate client UI is needed while Euroscope has focus.
- Acceptable p95 latency and false-execution rate, and whether ambiguous
  commands require confirmation or always request repetition.
- Exact first-release command subset and eligible-aircraft rule, particularly
  where current handoff/frequency state is incomplete.
- Eligible model weights and license, local backend, minimum hardware, and
  whether a verified smaller-model fallback is desirable.
- How the proxy is securely paired with its FSD controller session, and
  whether remote single-player connections require TLS in the application or
  a documented trusted tunnel.
