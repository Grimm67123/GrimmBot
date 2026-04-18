# 🐺 GrimmBot

⚠️ **LICENSE NOTICE** ⚠️

**IN ORDER TO USE THIS SOFTWARE, YOU MUST READ, UNDERSTAND AND AGREE TO THE GNU AGPL V3 LICENSE INCLUDED IN THIS REPOSITORY.**

---

## What is GrimmBot?

GrimmBot is an autonomous AI agent that runs inside a sandboxed Docker container with full desktop and browser control. It learns from its mistakes, builds its own tools, and gets better over time.

### Core Features

- **🧠 Self-Learning from Mistakes** — When the agent hits an error (blocked command, failed action, wrong approach), it writes a rule to `data/adaptation.json` so it never repeats the same mistake. Rules are retrieved using keyword matching — only rules relevant to the current task are loaded.
- **🛠️ Custom Tool Creation** — If the agent's built-in tools can't handle a task, it writes and registers a new Python tool on the fly. Custom tools persist across sessions in `data/custom_tools/`.
- **👀 Zero-Token Monitoring**: Grimmbot bypasses expensive LLM polling by dynamically writing custom Python tools that monitor changes locally, saving API tokens and requests during long-running observation tasks.
- **💾 Persistent RAG Memory** — The agent remembers past tasks and outcomes using TF-IDF semantic search. It retrieves relevant past experiences to inform future actions (`data/memory.json`).
- **⏰ Task Scheduling** — Supports one-time, daily, and interval-based background task scheduling with disk persistence.
- **👍 Feedback & Dataset Creation** — Rate agent responses with thumbs up/down and export a JSONL dataset for fine-tuning by pressing the button in the UI. **Feedback is restricted to local models only** (Ollama, LM Studio) — the buttons are hidden when using cloud providers.
- **🌐 Browser Automation** — Full Chromium control via CDP with DOM extraction. The agent reads webpage structure through tagged interactive elements, not screenshots.
- **🔒 Sandboxed Execution** — All commands run inside Docker. Domain allowlists, command allowlists, and user approval gates prevent unintended actions.
- **👀 Live Desktop View** — Watch the agent work in real-time through the noVNC panel in the web interface.

---

### Architecture

| Module | Purpose |
|---|---|
| `core.py` | Configuration, safety rules, custom tool registry, system prompts, and tool definitions |
| `agent.py` | Main agent loop, LLM communication, tool dispatch |
| `tools.py` | 40+ built-in tools (shell, file I/O, browser, memory, scheduling) |
| `memory.py` | RAG memory store, keyword-based adaptation retrieval, RLHF feedback store |
| `scheduler.py` | Persistent background task scheduler |
| `screen.py` | Chromium CDP control, DOM extraction, mouse/keyboard input |
| `grimmbot.py` | FastAPI web server, WebSocket handlers, command routing |

---

## Setup

**Prerequisites:** Docker and Docker Compose installed.

### 1. Clone and configure

```bash
git clone https://github.com/Grimm67123/GrimmBot.git
cd directoryforthegrimmbotfolder
cp .env.example .env
```

Edit `.env` with your model and API key:

```
# Local model example (Ollama)
LLM_MODEL=ollama/gemma3:4b
OLLAMA_API_BASE=http://host.docker.internal:11434

# Cloud model example (Gemini)
LLM_MODEL=gemini/gemini-2.5-flash
PROVIDER_API_KEY=your-api-key-here
```


Set your security allowlists:

```env
ALLOWED_DOMAINS=github.com,wikipedia.org
ALLOWED_COMMANDS=ls,cat,echo,grep,find,python3,pip,node,npm,git,curl,wget
```

### 2. Start

```bash
docker compose up --build
```

### 3. Use

Open `http://localhost:5000` in your browser in order to use the agent.

---

## Gallery

>GrimmBot's webUI at localhost:5000
<img width="1363" height="637" alt="Image" src="https://github.com/user-attachments/assets/8a8ea1b5-5168-450f-aea1-217f775fe99a" />

>GrimmBot improves itself by creating a rule that makes it avoid making the same mistake again.
<img width="1356" height="634" alt="Image" src="https://github.com/user-attachments/assets/2d934971-a5b0-4dee-aa14-c6614adebae0" />

>Custom tool creation approval prompt.
<img width="1363" height="643" alt="Image" src="https://github.com/user-attachments/assets/09bf2ab8-8927-4d37-a0c7-a352eb64393f" />

---

## Commands

Type these in the chat (no LLM call required):

| Command | What it does |
|---|---|
| `!help` | Show all commands |
| `!verbose` | Toggle detailed tool/thought logs |
| `!throttle [N/off]` | Delay between agent actions |
| `!commssafeguard` | Toggle approval for typing/clicking actions |
| `!login <url> [profile]` | Open browser for manual login via VNC |
| `!profiles` | List Chromium profiles |
| `!memory [show/reset]` | View or clear RAG memory |
| `!schedule` | View scheduled tasks |
| `!files` | List files in the wormhole |
| `!tools` | List built-in tools |
| `!custom-tools` | List agent-created tools |
| `!emergency` | Halt all agent operations |

---

## Portability

All learning is stored in the `data/` folder and is portable between machines:

- **`data/adaptation.json`** — Self-learned rules. Copy to another instance to transfer learned behaviors.
- **`data/memory.json`** — RAG memory (per-profile). Preserves context and knowledge.
- **`data/custom_tools/`** — Agent-created Python tools. Fully portable.
- **`data/scheduler/`** — Scheduled task definitions.
- **`data/feedback.json`** — Thumbs up/down feedback data (local models only).

---

## FAQ

**Which LLMs work?** Any model supported by [LiteLLM](https://github.com/BerriAI/litellm) — Gemini, GPT, Claude, Ollama, LM Studio, Groq, Mistral, and others.

**Can I use fully local/offline models?** Yes. For example, for Ollama, set `LLM_MODEL=ollama/your-model` and GrimmBot connects to your host's Ollama via `host.docker.internal`. No internet needed for the agent itself. Right now, only Ollama, LocalAI, text gen, vLLM and LM Studio are supported.

**How does the wormhole work?** The `wormhole/` folder is shared between your host and the container. Place files there for the agent to access, or retrieve files the agent creates.

**What happens when the agent encounters an error?** It analyzes the error, writes a rule to `adaptation.json` to avoid it in the future, and retries. Over time, the agent accumulates rules that make it more reliable.

**How do I run tests?**
```bash
pip install pytest
cd directoryforthegrimmbotfolder
pytest tests/ -v
```

---

### Contribution Policy

Sorry, this project currently does not accept external pull requests. Bug reports and feature requests via GitHub Issues are welcome.

---

⭐ **If this project impressed you, consider leaving a star!**
