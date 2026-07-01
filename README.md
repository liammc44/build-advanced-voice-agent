# Build an advanced voice agent

A voice agent you can talk to and think out loud with — built on [AssemblyAI][aai] real-time
speech-to-text (Universal-3.5 Pro), with Claude as the brain and Cartesia for the voice.

![Voice agent](assets/voice-agent-thumbnail.webp)

Run it as a brainstorming partner that pushes back on your ideas, or an interview coach that drills
you on STAR answers — and switch between them by voice, mid-conversation.

```
mic → AssemblyAI Streaming STT (Universal-3.5 Pro) → Claude → Cartesia → speakers
```

It takes three keys — one each for [AssemblyAI][aai] (speech-to-text),
[Anthropic](https://console.anthropic.com/settings/keys) (Claude), and
[Cartesia](https://play.cartesia.ai/keys) (voice). Copy `.env.example` to `.env` and paste them in.

## `coach.py` — terminal voice agent

A single Python file. Streams your microphone to AssemblyAI's Universal Streaming STT, routes the
transcript through Claude in character, and speaks the reply back with Cartesia. Pick a mode at
launch, or switch by voice mid-conversation ("switch to interview mode").

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install websockets sounddevice numpy requests anthropic python-dotenv
cp .env.example .env          # then paste your three keys into .env
python coach.py               # or: python coach.py interview
```

Run it with headphones — a terminal has no echo cancellation.

## `web/index.html` — browser voice agent

A single, self-contained HTML page with a Material 3 interface. The same two modes, each with its
own colour palette, plus a live transcript, voice mode-switching, and a **context panel** showing
what the coach picks up as you talk. Browsers do echo cancellation for you, so no headphones needed.

```bash
python3 -m http.server 8000 --directory web
# open http://localhost:8000 in Chrome or Edge, then paste your three keys in Settings (⚙)
```

## Full setup guide

To get the step-by-step instructions, including how to get your API keys, read the full guide at:

**https://loopnews.io/academy/build-advanced-voice-agent**

[aai]: https://www.assemblyai.com/docs/voice-agents/voice-agent-api?utm_source=newsletter&utm_medium=influencer&utm_campaign=loop&utm_content=u35_realtime
