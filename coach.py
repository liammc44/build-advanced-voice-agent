#!/usr/bin/env python3
"""
A single-file Python voice agent on AssemblyAI's Voice Agent API, with two modes:

  * interview  — behavioral interview prep, coaching you through STAR answers
  * brainstorm — a brainstorming partner that pushes back and digs deeper

You can switch modes mid-conversation just by asking out loud — say something like
"switch to brainstorming" or "switch to interview mode" and it changes character on the fly.

One WebSocket does everything: your microphone audio streams up and is transcribed by
Universal-3.5 Pro Realtime (the default speech foundation under the Voice Agent API since
2026-06-23), the managed LLM answers in character, and the spoken reply streams back and
plays through your speakers. No separate STT / LLM / TTS wiring.

    pip install websockets sounddevice numpy python-dotenv
    export ASSEMBLYAI_API_KEY=your_key       # or drop it in a .env file next to this script

    python coach.py interview      # start in interview prep
    python coach.py brainstorm     # start in brainstorming partner
    python coach.py                # asks which one

WEAR HEADPHONES. A terminal app has no OS-level acoustic echo cancellation, so without them
your mic hears the agent's own voice and it interrupts itself. Headphones fix it completely.

Press Ctrl+C to end the session cleanly (this stops billing immediately).

Notes on the two model choices in the prompt:
  * Transcription model — the Voice Agent API does not expose a per-session speech-model knob;
    Universal-3.5 Pro Realtime is already the platform default, so transcription runs on it.
  * Response LLM — likewise not selectable through the public schema. The conversation is run
    by AssemblyAI's managed LLM; the SYSTEM_PROMPT for each mode is what gives it its character.
"""

import asyncio
import base64
import json
import os
import re
import signal
import sys

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

# ── Connection ───────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
WS_URL = "wss://agents.assemblyai.com/v1/ws"

# ── Audio (the Voice Agent API speaks and listens in PCM16 mono @ 24 kHz) ─────
SAMPLE_RATE = 24_000
BLOCKSIZE = 1200            # 50 ms of audio per chunk

# ── Agent voice (browse more at .../voice-agent-api/voices) ──────────────────
VOICE = "ivy"

# Patient turn-taking: people pause mid-thought and need room, so we wait longer before
# deciding the turn is over. interrupt_response lets you barge in to cut the agent off.
TURN_DETECTION = {
    "vad_threshold": 0.5,
    "min_silence": 1800,
    "max_silence": 6000,
    "interrupt_response": True,
}

# Both prompts are voice-tuned: identity first, no markdown, one thing per turn, explicit
# permissions, and the exact filler phrases to avoid. See .../voice-agent-api/prompting-guide.

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
    "interview": {
        "name": "interview prep",
        "greeting": (
            "Hey, I'm your interview coach. Tell me the role you're prepping for, "
            "and we'll run through some behavioral questions together. You can also say "
            "switch to brainstorming anytime."
        ),
        "system_prompt": INTERVIEW_PROMPT,
    },
    "brainstorm": {
        "name": "brainstorming partner",
        "greeting": (
            "Hey, I'm your brainstorming partner. What's on your mind — a business idea, "
            "a trip, a plan? Tell me what you want to dig into. You can also say "
            "switch to interview mode anytime."
        ),
        "system_prompt": BRAINSTORM_PROMPT,
    },
}

# Accept a few friendly spellings for each mode, from the CLI or the startup prompt.
ALIASES = {
    "interview": "interview", "interviews": "interview", "prep": "interview", "i": "interview", "1": "interview",
    "brainstorm": "brainstorm", "brainstorming": "brainstorm", "ideas": "brainstorm", "b": "brainstorm", "2": "brainstorm",
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
        print("Pick a mode:  [1] interview prep   [2] brainstorming partner")
        key = ALIASES.get(input("> ").strip().lower())
    return key


async def main(mode_key: str) -> None:
    if not API_KEY:
        sys.exit("Set ASSEMBLYAI_API_KEY (env var or a .env file) and try again.")

    current = mode_key
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    ready = asyncio.Event()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop.set)
        loop.add_signal_handler(signal.SIGTERM, stop.set)
    except NotImplementedError:
        pass  # non-Unix: Ctrl+C still raises KeyboardInterrupt, handled in __main__

    speaker = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    speaker.start()

    def flush_playback() -> None:
        """Drop any audio still queued for the speaker (used on barge-in)."""
        speaker.abort()
        speaker.start()

    print(f"Connecting to the Voice Agent API…  mode: {MODES[current]['name']}  (wear headphones!)\n")
    try:
        async with websockets.connect(
            WS_URL, additional_headers={"Authorization": f"Bearer {API_KEY}"}
        ) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": MODES[current]["system_prompt"],
                    "greeting": MODES[current]["greeting"],
                    "output": {"voice": VOICE},
                    "input": {"turn_detection": TURN_DETECTION},
                },
            }))

            async def send_audio() -> None:
                # Don't stream a single frame before session.ready, or it's dropped.
                await ready.wait()

                def on_audio(indata, _frames, _time, _status) -> None:
                    loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                    blocksize=BLOCKSIZE, callback=on_audio):
                    while True:
                        chunk = await mic_queue.get()
                        await ws.send(json.dumps({
                            "type": "input.audio",
                            "audio": base64.b64encode(chunk).decode(),
                        }))

            async def switch_mode(target: str) -> None:
                nonlocal current
                current = target
                # system_prompt is mutable mid-session; resend only the field we're changing.
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {"system_prompt": MODES[target]["system_prompt"]},
                }))
                print(f"\n— switched to {MODES[target]['name']} —\n")

            async def receive_events() -> None:
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")

                    if kind == "session.ready":
                        ready.set()
                        print('● Listening — start talking. Say "switch to brainstorming" or '
                              '"switch to interview mode" anytime. Ctrl+C to end.\n')

                    elif kind == "reply.audio":
                        pcm = np.frombuffer(base64.b64decode(event["data"]), dtype=np.int16)
                        speaker.write(pcm)

                    elif kind == "input.speech.started":
                        flush_playback()                      # snappy barge-in

                    elif kind == "reply.done" and event.get("status") == "interrupted":
                        flush_playback()                      # stop stale audio after a cut-off

                    elif kind == "transcript.user":
                        print(f"You:   {event['text']}")
                        target = detect_switch(event["text"], current)
                        if target:
                            await switch_mode(target)

                    elif kind == "transcript.agent":
                        print(f"Agent: {event['text']}\n")

                    elif kind == "session.error":
                        print(f"[session.error] {event.get('code')}: {event.get('message')}",
                              file=sys.stderr)

                    elif kind == "session.ended":
                        stop.set()

            workers = [asyncio.create_task(send_audio()), asyncio.create_task(receive_events())]
            stopper = asyncio.create_task(stop.wait())

            # Run until a worker exits (e.g. server closed the socket) or Ctrl+C / session.ended.
            await asyncio.wait([*workers, stopper], return_when=asyncio.FIRST_COMPLETED)

            # Tell the server we're done so it doesn't hold a billable 30-second resume window.
            try:
                await ws.send(json.dumps({"type": "session.end"}))
                await asyncio.wait_for(ws.wait_closed(), timeout=2)
            except Exception:
                pass

            for task in (*workers, stopper):
                task.cancel()
            await asyncio.gather(*workers, stopper, return_exceptions=True)

        print("\nSession ended.")
    finally:
        speaker.stop()
        speaker.close()


if __name__ == "__main__":
    selected = choose_mode()
    try:
        asyncio.run(main(selected))
    except KeyboardInterrupt:
        print("\nSession ended.")
