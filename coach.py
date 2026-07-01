#!/usr/bin/env python3
"""
A single-file Python voice coach. Your microphone streams to AssemblyAI's Universal
Streaming STT (Universal-3.5 Pro), the transcript is answered by Claude in character, and
the spoken reply is synthesised by Cartesia and played back — the whole pipeline wired by
hand, no framework. Two modes you can switch between mid-conversation just by asking out loud:

  * brainstorm — a brainstorming partner that pushes back and digs deeper (default)
  * interview  — behavioral interview prep, coaching you through STAR answers

Say something like "switch to interview mode" or "switch to brainstorming" and it changes
character on the fly.

    pip install websockets sounddevice numpy requests anthropic python-dotenv

Three keys — drop them in a .env file next to this script (see .env.example):

    ASSEMBLYAI_API_KEY   real-time speech-to-text
    ANTHROPIC_API_KEY    Claude, the coach's brain
    CARTESIA_API_KEY     text-to-speech voice

    python coach.py brainstorm     # start in brainstorming partner
    python coach.py interview      # start in interview prep
    python coach.py                # asks which one

The pipeline (mirrors web/index.html):

    mic ─► AssemblyAI Streaming STT (Universal-3.5 Pro, voice_focus) ─► Claude ─► Cartesia ─► speakers

Each turn primes the STT model with the coach's last reply via `agent_context`, so a mumbled
"user at gmail dot com" resolves to user@gmail.com on the next turn.

WEAR HEADPHONES. A terminal app has no OS-level acoustic echo cancellation, so without them
your mic hears the coach's own voice and it interrupts itself. Headphones fix it completely.
(There's also a self-echo guard that drops any "user" turn that is mostly the coach's own
words, but headphones are the real fix.)

Press Ctrl+C to end the session cleanly.
"""

import asyncio
import json
import os
import re
import signal
import sys
import urllib.parse

import numpy as np
import requests
import sounddevice as sd
import websockets
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

# ── Keys (three separate services now — this is not the all-in-one Voice Agent API) ──────────
AAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
CARTESIA_KEY = os.environ.get("CARTESIA_API_KEY")

# ── 1) Speech-to-text: AssemblyAI Universal Streaming v3 ─────────────────────────────────────
AAI_WS = "wss://streaming.assemblyai.com/v3/ws"
SPEECH_MODEL = "universal-3-5-pro"       # the brief's model
VOICE_FOCUS = "near-field"               # near-field (headset/phone) | far-field (room)
FORMAT_TURNS = True                      # punctuation/casing/ITN + email normalisation on finals
KEYTERMS: list[str] = []                 # e.g. ["AssemblyAI", "Cartesia"] — boosts rare words
MIN_TURN_SILENCE = 560                   # ms before a pause counts as end-of-turn (default 400)
MAX_TURN_SILENCE = 2400                  # ms hard cap (default 1280)
MIC_RATE = 16000                         # PCM16 mono up to AssemblyAI
FRAME_MS = 100                           # 100 ms per frame (AssemblyAI wants 50–1000 ms chunks)
FRAME = MIC_RATE * FRAME_MS // 1000

# ── 2) The brain: Claude ─────────────────────────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-6"          # good latency/quality for voice; opus-4-8 smarter/slower
REPLY_MAX_TOKENS = 300                   # replies are two or three spoken sentences
MAX_TURNS = 16                           # rolling context window: keep the last N exchanges for Claude

# ── 3) The voice: Cartesia (Sonic) TTS ───────────────────────────────────────────────────────
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_MODEL = "sonic-3.5"
CARTESIA_VERSION = "2026-03-01"
CARTESIA_VOICE = "b7d50908-b17c-442d-ad8d-810c63997ed9"   # "California Girl" — US female
TTS_RATE = 24000                         # Cartesia returns raw pcm_f32le @ 24 kHz

# ── Prompts ──────────────────────────────────────────────────────────────────────────────────
# Both are voice-tuned: identity first, no markdown, one thing per turn, explicit permissions,
# and the exact filler phrases to avoid.

INTERVIEW_PROMPT = """\
You are a behavioral interview coach. You run realistic practice for questions like "Tell me \
about a time you handled conflict," and you coach the candidate to answer using the STAR method: \
situation, task, action, result.

Run it like a real interview with a coach sitting beside it. Ask one behavioral question, let the \
candidate answer all the way through, then give short, specific feedback before the next question. \
Never stack two questions in one turn.

Keep every turn to two or three sentences, because this is spoken out loud. When you give feedback, \
name one concrete thing that worked and one concrete thing to fix, both tied to what they actually \
said, not generic advice.

Listen for the STAR pieces. If they skip the result, or never separate what they personally did \
from what the team did, ask them to fill that gap. Push for specifics: their exact role, real \
numbers, the outcome.

Be warm and direct. Encourage them, but stay honest. If an answer rambled or stayed vague, say so \
plainly and tell them how to tighten it. React like a person would: "good, strong opening," or \
"okay, but you never told me what you did."

Speak in plain spoken sentences. Round any numbers. Never say "great question," "happy to help," \
or "as an AI." No lists read aloud, no markdown, no exclamation marks.

If this is the very start of the conversation, ask which role or kind of interview they're \
preparing for, then ask your first behavioral question. If you are mid-conversation and have just \
switched into this mode, say one short line to mark the shift, then ask your first question.\
"""

BRAINSTORM_PROMPT = """\
You are a sharp brainstorming partner. You help the person explore an idea in more depth than they \
would on their own: business ideas, personal projects, trips, plans, whatever they bring.

Think with them, do not just agree. Build on what they say, then push: ask the question they are \
avoiding, surface a tradeoff they skipped, offer an angle they have not considered. Honest pushback \
is the point. Be candid when something is weak, but stay constructive and keep the energy up.

Keep every turn to two or three sentences, because this is spoken out loud. Move one thought at a \
time, and usually end on a single pointed question that takes the idea somewhere new, rather than \
piling on three questions at once.

Go for depth over breadth. When they float an idea, get concrete fast: who is it for, what would it \
actually take, what is the first real test, what would break it. Offer your own ideas and opinions \
freely. Suggest concrete options, not vague encouragement.

Be warm but honest. If an idea is vague or the reasoning is thin, say so plainly and help sharpen \
it. React like a person would: "okay, I like that," or "hold on, that part is the hard bit, how \
would it actually work."

Speak in plain spoken sentences. Round any numbers. Never say "great idea," "happy to help," or \
"as an AI." No lists read aloud, no markdown, no exclamation marks.

If this is the very start of the conversation, ask what they want to brainstorm, then dig in. If \
you are mid-conversation and have just switched into this mode, say one short line to mark the \
shift, then ask what they want to dig into.\
"""

MODES = {
    "brainstorm": {
        "name": "brainstorming partner",
        "greeting": (
            "Hey, I'm your brainstorming partner. What's on your mind — a business idea, "
            "a trip, a plan? Tell me what you want to dig into. You can also say "
            "switch to interview mode anytime."
        ),
        "system_prompt": BRAINSTORM_PROMPT,
    },
    "interview": {
        "name": "interview prep",
        "greeting": (
            "Hey, I'm your interview coach. Tell me the role you're prepping for, "
            "and we'll run through some behavioral questions together. You can also say "
            "switch to brainstorming anytime."
        ),
        "system_prompt": INTERVIEW_PROMPT,
    },
}

# Accept a few friendly spellings for each mode, from the CLI or the startup prompt.
ALIASES = {
    "brainstorm": "brainstorm", "brainstorming": "brainstorm", "ideas": "brainstorm", "b": "brainstorm", "1": "brainstorm",
    "interview": "interview", "interviews": "interview", "prep": "interview", "i": "interview", "2": "interview",
}

# Mid-session switching. We only switch on an explicit command — a switch verb followed by "to"
# and the target mode word, or "<mode> mode/prep" — so an interview story that merely mentions
# "brainstorm" (or vice-versa) doesn't flip modes by accident.
_VERB = r"(?:switch|change|swap|flip|move|go|jump|put|take)\w*"
SWITCH_TO = {
    "brainstorm": re.compile(
        rf"\b{_VERB}\b[\w' ]{{0,15}}?\bto\b[\w' ]{{0,15}}?\bbrainstorm"
        r"|\bbrainstorm(?:ing)?\s+mode\b", re.I),
    "interview": re.compile(
        rf"\b{_VERB}\b[\w' ]{{0,15}}?\bto\b[\w' ]{{0,15}}?\binterview"
        r"|\binterview\s+(?:prep|practice|mode)\b", re.I),
}


def detect_switch(text: str, current: str) -> str | None:
    """Return the mode to switch to if `text` is an explicit switch command, else None."""
    for target, pattern in SWITCH_TO.items():
        if target != current and pattern.search(text):
            return target
    return None


def choose_mode() -> str:
    """Pick a starting mode key from argv, falling back to a one-line interactive prompt."""
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    key = ALIASES.get(arg)
    while key is None:
        print("Pick a mode:  [1] brainstorming partner   [2] interview prep")
        try:
            key = ALIASES.get(input("> ").strip().lower())
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nNo mode selected.")
    return key


class Coach:
    """Owns the session: one STT WebSocket, one Claude client, and the barge-in state machine."""

    def __init__(self, mode_key: str) -> None:
        self.current = mode_key
        self.anthropic = AsyncAnthropic(api_key=ANTHROPIC_KEY)

        self.loop: asyncio.AbstractEventLoop | None = None
        self.stt_ws = None                               # the STT WebSocket, set in run()
        self.out_q: asyncio.Queue = asyncio.Queue()      # outgoing to STT: bytes (audio) or dict (control)
        self.stop_evt = asyncio.Event()
        self.begin_evt = asyncio.Event()                 # set when STT sends Begin (connect watchdog)

        self.history: list[dict] = []       # Claude turns: alternating user/assistant, starting with user
        self.stt_ready = False
        self.speaking = False
        self.epoch = 0                       # bumped on every turn / barge-in; stale async work checks it
        self.last_agent_reply = ""           # used to spot the coach transcribing its own voice
        self.triggered_order = -1            # the turn we've already answered
        self.barged_order = -1               # most recent turn that interrupted the coach (echo suspect)
        self.active: asyncio.Task | None = None      # the in-flight greet()/respond() task
        self.pending_final: asyncio.TimerHandle | None = None

    def system_prompt(self) -> str:
        return MODES[self.current]["system_prompt"]

    # ── Task plumbing ────────────────────────────────────────────────────────────────────────
    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        task.add_done_callback(self._on_task_done)
        return task

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            print(f"\n[error] {exc}", file=sys.stderr)

    # ── Speech-to-text ───────────────────────────────────────────────────────────────────────
    def stt_url(self) -> str:
        params = [
            ("token", AAI_KEY),                       # raw key works directly as the token
            ("sample_rate", str(MIC_RATE)),
            ("encoding", "pcm_s16le"),
            ("format_turns", "true" if FORMAT_TURNS else "false"),
            ("speech_model", SPEECH_MODEL),
            ("voice_focus", VOICE_FOCUS),
            ("min_turn_silence", str(MIN_TURN_SILENCE)),
            ("max_turn_silence", str(MAX_TURN_SILENCE)),
        ]
        for kt in KEYTERMS:                           # repeat the key once per term
            params.append(("keyterms_prompt", kt))
        return AAI_WS + "?" + urllib.parse.urlencode(params)

    def _mic_cb(self, indata, _frames, _time, _status) -> None:
        # Runs on sounddevice's own thread. Hand raw PCM16 bytes to the event loop.
        if self.stt_ready and self.loop is not None:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, bytes(indata))

    async def sender(self) -> None:
        """Single writer for the STT socket — serialises audio frames and control messages."""
        while True:
            item = await self.out_q.get()
            if self.stt_ws is None:
                continue
            try:
                if isinstance(item, (bytes, bytearray)):
                    if self.stt_ready:
                        await self.stt_ws.send(item)              # binary frame → audio
                else:
                    await self.stt_ws.send(json.dumps(item))     # JSON text → control
            except Exception:
                break

    async def receiver(self) -> None:
        """Read STT events and drive the turn state machine."""
        try:
            async for raw in self.stt_ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                kind = m.get("type")
                if kind == "Begin":
                    self.stt_ready = True
                    self.begin_evt.set()
                    print('● Listening — just talk. Say "switch to interview mode" or '
                          '"switch to brainstorming" anytime. Ctrl+C to end.\n')
                    self.active = self._spawn(self.greet())
                elif kind == "Turn":
                    self.handle_turn(m)
                elif kind == "Error":
                    print(f"\n[assemblyai] {m.get('error') or m.get('error_code')}", file=sys.stderr)
                elif kind == "Termination":
                    break
        finally:
            self.stop_evt.set()

    def handle_turn(self, m: dict) -> None:
        text = (m.get("transcript") or "").strip()
        if not text:
            return
        order = m.get("turn_order")

        # Barge-in only on genuinely NEW user speech (not the trailing formatted frame of the turn
        # we just answered), and only while the coach is producing output.
        producing = self.speaking or (self.active is not None and not self.active.done())
        if producing and order != self.triggered_order:
            self.barged_order = order                 # interrupted the coach → an echo suspect
            self.barge_in()

        if m.get("end_of_turn") and order != self.triggered_order:
            # With format_turns on, an unformatted end_of_turn arrives first, then a formatted one
            # for the same turn_order. Finalize on the formatted text, but never hang waiting.
            self._cancel_pending_final()
            if m.get("turn_is_formatted") or not FORMAT_TURNS:
                self.finalize_turn(order, text)
            else:
                self.pending_final = self.loop.call_later(0.7, self.finalize_turn, order, text)

    def _cancel_pending_final(self) -> None:
        if self.pending_final is not None:
            self.pending_final.cancel()
            self.pending_final = None

    def finalize_turn(self, order: int, text: str) -> None:
        if order == self.triggered_order:            # formatted turn already beat the fallback timer
            return
        self.triggered_order = order
        self.pending_final = None
        # Only guard against self-echo for a turn that interrupted the coach — a genuine answer given
        # after the coach has finished speaking may legitimately reuse the question's words.
        if order == self.barged_order and self.is_self_echo(text):
            return
        print(f"You:   {text}")
        target = detect_switch(text, self.current)
        if target:
            self.switch_mode(target)
        self.speaking = True                         # claim output immediately (closes the barge-in gap)
        self.active = self._spawn(self.respond(text))

    # Guard against weak echo-cancellation transcribing the coach's own voice as a user turn.
    @staticmethod
    def _norm(s: str) -> list[str]:
        return [w for w in re.sub(r"[^a-z0-9 ]", "", s.lower()).split() if w]

    def is_self_echo(self, text: str) -> bool:
        if not self.last_agent_reply:
            return False
        words = self._norm(text)
        if len(words) < 3:
            return False
        spoken = set(self._norm(self.last_agent_reply))
        if len(spoken) < 6:                         # a short coach reply is mostly function words —
            return False                            # overlap there is not evidence of an echo
        hits = sum(1 for w in words if w in spoken)
        return hits >= 4 and hits / len(words) > 0.6   # need real substance, not 3 shared function words

    def switch_mode(self, target: str) -> None:
        self.current = target
        print(f"\n— switched to {MODES[target]['name']} —\n")   # next reply uses the new system prompt

    def set_agent_context(self, text: str) -> None:
        # Prime the STT model with the coach's last question (the agent_context demo).
        if self.stt_ready:
            self.out_q.put_nowait({"type": "UpdateConfiguration", "agent_context": text[:600]})

    # ── The brain + the voice ────────────────────────────────────────────────────────────────
    async def greet(self) -> None:
        """The coach opens the conversation (spoken only; not in Claude history, because the
        Anthropic messages array must start with a user turn)."""
        my = self.epoch
        greeting = MODES[self.current]["greeting"]
        print(f"Coach: {greeting}\n")
        self.last_agent_reply = greeting
        self.speaking = True
        self.set_agent_context(greeting)
        try:
            await self.say(greeting, my)
        finally:
            if my == self.epoch:
                self.speaking = False

    async def respond(self, user_text: str) -> None:
        """One full turn: user said `user_text` → Claude replies (streamed) → speak it."""
        my = self.epoch
        self.history.append({"role": "user", "content": user_text})
        while len(self.history) > 2 * MAX_TURNS:      # rolling window; drop whole (user, assistant) pairs
            del self.history[:2]                      # from the front, so the list stays user-first
        mark = len(self.history) - 1                  # index of the speculative user turn (for rollback)
        reply = ""
        try:
            print("Coach: ", end="", flush=True)
            async with self.anthropic.messages.stream(
                model=LLM_MODEL,
                max_tokens=REPLY_MAX_TOKENS,
                system=self.system_prompt(),
                messages=self.history,
            ) as stream:
                async for delta in stream.text_stream:
                    print(delta, end="", flush=True)
                    reply += delta
            print("\n")

            reply = reply.strip()
            if not reply:                             # nothing to say — undo the speculative user turn
                del self.history[mark:]
                return
            self.history.append({"role": "assistant", "content": reply})
            self.last_agent_reply = reply
            self.set_agent_context(reply)             # prime STT for the next user turn
            try:
                await self.say(reply, my)             # a playback error must NOT roll back the committed reply
            except Exception as e:
                print(f"\n[tts error] {e}", file=sys.stderr)
        except asyncio.CancelledError:
            print()                                   # end the half-streamed line cleanly
            # If we were barged before the reply committed, roll the speculative user turn back.
            if len(self.history) > mark and self.history[-1]["role"] == "user":
                del self.history[mark:]
            raise
        except Exception as e:
            del self.history[mark:]
            print(f"\n[claude error] {e}", file=sys.stderr)
        finally:
            if my == self.epoch:
                self.speaking = False

    async def say(self, text: str, my: int) -> None:
        """Synthesise `text` with Cartesia and play it, interruptibly."""
        if my != self.epoch:
            return
        audio = await asyncio.to_thread(self.synthesize, text)
        if audio is None or my != self.epoch:
            return
        await asyncio.to_thread(self._play_blocking, audio, my)

    def synthesize(self, text: str):
        """Cartesia /tts/bytes → raw float32 PCM @ 24 kHz. Returns a numpy array, or None."""
        try:
            r = requests.post(
                CARTESIA_URL,
                headers={
                    "Authorization": f"Bearer {CARTESIA_KEY}",
                    "Cartesia-Version": CARTESIA_VERSION,
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": CARTESIA_MODEL,
                    "transcript": text,
                    "voice": {"mode": "id", "id": CARTESIA_VOICE},
                    "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": TTS_RATE},
                },
                timeout=(5, 15),   # (connect, read) — bounds shutdown if Cartesia stalls
            )
        except Exception as e:
            print(f"\n[cartesia error] {e}", file=sys.stderr)
            return None
        if r.status_code != 200:
            print(f"\n[cartesia {r.status_code}] {r.text[:200]}", file=sys.stderr)
            return None
        try:                                          # tolerate a truncated / non-PCM 200 body
            content = r.content
            n = len(content) - (len(content) % 4)     # drop any trailing partial float32
            audio = np.frombuffer(content[:n], dtype=np.float32) if n else np.empty(0, np.float32)
        except Exception as e:
            print(f"\n[cartesia decode] {e}", file=sys.stderr)
            return None
        return audio if audio.size else None

    def _play_blocking(self, audio, my: int) -> None:
        # Play in ~100 ms blocks so a barge-in (which bumps self.epoch) stops us within one block —
        # even if this worker only reached playback *after* barge_in() already ran. Owning the stream
        # here (rather than sd.play/sd.stop) avoids the race where sd.stop() fires before playback starts.
        block = TTS_RATE // 10
        out = sd.OutputStream(samplerate=TTS_RATE, channels=1, dtype="float32")
        out.start()
        try:
            for i in range(0, len(audio), block):
                if my != self.epoch:                  # barged in: discard buffered audio immediately
                    out.abort()
                    return
                out.write(audio[i:i + block].reshape(-1, 1))
            if my == self.epoch:
                out.stop()                            # normal end: drain the buffer so the tail isn't clipped
            else:
                out.abort()
        finally:
            out.close()

    def barge_in(self) -> None:
        """User started talking over the coach: invalidate the in-flight reply and cut the audio."""
        self.epoch += 1                               # stale greet()/respond()/say()/_play_blocking() bail on this
        if self.active is not None and not self.active.done():
            self.active.cancel()
        self.speaking = False

    # ── Lifecycle ────────────────────────────────────────────────────────────────────────────
    async def run(self) -> None:
        if not (AAI_KEY and ANTHROPIC_KEY and CARTESIA_KEY):
            missing = [n for n, v in (("ASSEMBLYAI_API_KEY", AAI_KEY),
                                      ("ANTHROPIC_API_KEY", ANTHROPIC_KEY),
                                      ("CARTESIA_API_KEY", CARTESIA_KEY)) if not v]
            sys.exit(f"Missing key(s): {', '.join(missing)}. Set them (env var or .env) and try again.")

        self.loop = asyncio.get_running_loop()

        def request_stop() -> None:
            # First Ctrl+C: begin a graceful shutdown and hand SIGINT back to Python, so a second
            # Ctrl+C raises KeyboardInterrupt and force-quits if shutdown ever wedges.
            try:
                self.loop.remove_signal_handler(signal.SIGINT)
            except Exception:
                pass
            self.stop_evt.set()

        try:
            self.loop.add_signal_handler(signal.SIGINT, request_stop)
            self.loop.add_signal_handler(signal.SIGTERM, self.stop_evt.set)
        except NotImplementedError:
            pass  # non-Unix: Ctrl+C still raises KeyboardInterrupt, handled in __main__

        print(f"Connecting to AssemblyAI…  mode: {MODES[self.current]['name']}  (wear headphones!)\n")
        async with websockets.connect(self.stt_url(), max_size=None) as ws:
            self.stt_ws = ws
            with sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                                   blocksize=FRAME, callback=self._mic_cb):
                recv = self._spawn(self.receiver())
                send = self._spawn(self.sender())

                # Don't sit on "Connecting…" forever: proceed on Begin, but bail on an 8s timeout
                # (half-open socket) or as soon as the receiver reports the socket closed (bad key).
                begin = asyncio.ensure_future(self.begin_evt.wait())
                closed = asyncio.ensure_future(self.stop_evt.wait())
                done, pending = await asyncio.wait(
                    {begin, closed}, timeout=8, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                if begin not in done and not self.stop_evt.is_set():
                    print("Couldn't reach AssemblyAI — check your key and network.", file=sys.stderr)
                    self.stop_evt.set()

                await self.stop_evt.wait()            # Ctrl+C, server Termination, or socket close

                # Tell the server we're done so it doesn't hold a billable resume window.
                try:
                    await ws.send(json.dumps({"type": "Terminate"}))
                except Exception:
                    pass
                self.barge_in()                       # cancel any in-flight reply + stop audio
                tasks = [t for t in (recv, send, self.active) if t is not None]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        print("\nSession ended.")


async def main(mode_key: str) -> None:
    await Coach(mode_key).run()


if __name__ == "__main__":
    selected = choose_mode()
    try:
        asyncio.run(main(selected))
    except KeyboardInterrupt:
        print("\nSession ended.")
