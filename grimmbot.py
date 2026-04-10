#!/usr/bin/env python3
"""
GrimmBot — Web Server Interface
"""

import os
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_DEBUG"] = "True"
import litellm
litellm.suppress_debug_info = True

import logging
import threading
import time
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from core import AgentConfig, TOOL_DEFINITIONS
from agent import GrimmAgent, TaskResult
from memory import get_memory, reset_memory, get_feedback_store
from scheduler import get_scheduler, ScheduledTask
from screen import (
    launch_chromium, close_chromium, list_chromium_profiles,
    wipe_chromium_profile, is_chromium_running, save_screenshot,
    CHROMIUM_PROFILE_DIR,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
for name in ["LiteLLM", "litellm", "httpx", "httpcore", "openai",
             "urllib3", "google", "memory", "scheduler", "screen"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

logger = logging.getLogger("grimmbot")
logger.setLevel(logging.INFO)

config = AgentConfig.from_env()
agent = GrimmAgent(config)

_local_api_prefixes = ["ollama", "lm_studio", "localai", "vllm", "text-gen"]
is_local_llm = bool(config.api_base and any(config.model.startswith(p) for p in _local_api_prefixes))


# ── Disclaimer Gate ──────────────────────────────────────────────────────────

DISCLAIMER_TEXT = """
<div style="max-width:720px;margin:0 auto;font-family:'Inter',system-ui,sans-serif;color:#e2e8f0;padding:2rem;">
<h1 style="color:#60a5fa;margin-bottom:0.5rem;">🐺 GrimmBot</h1>
<p style="color:#94a3b8;margin-bottom:1.5rem;">The self-improving sandboxed AI agent that adapts to failure.</p>

<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
<h2 style="color:#ffffff;margin-top:0;text-transform:uppercase;">📜 LICENSE</h2>
<p style="color:#ffffff;text-transform:uppercase;">THIS SOFTWARE IS LICENSED UNDER THE <strong>GNU AFFERO GENERAL PUBLIC LICENSE V3 (AGPL-3.0)</strong>. BY PROCEEDING, YOU CONFIRM YOU HAVE READ, UNDERSTOOD, AND AGREED TO THE <a href="https://www.gnu.org/licenses/agpl-3.0.en.html" target="_blank" style="color:#ffffff;text-decoration:underline;">AGPL-3.0 LICENSE</a> AND THE SECURITY NOTICE BELOW.</p>
</div>

<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
<h2 style="color:#f59e0b;margin-top:0;">⚠️ Security Notice</h2>
<p style="color:#fbbf24;"><strong>GrimmBot is alpha-stage software under active development. Expect bugs, unexpected behavior, and breaking changes.</strong></p>
<p>GrimmBot runs inside a Docker container and controls a sandboxed virtual desktop, not your full host computer. It can browse the web, execute approved commands, and create or modify files inside its allowed environment.</p>
<p><strong>Safety measures in place:</strong></p>
<ul style="color:#cbd5e1;margin-left:1.5rem;margin-top:0.5rem;">
<li>Isolation from the host system through Docker</li>
<li>Domain allowlist for browser navigation</li>
<li>Command allowlist for shell access</li>
<li>All agent outputs are placed in the wormhole directory (review before opening)</li>
<li>Approval system for higher-risk actions</li>
</ul>
<p style="margin-top:0.5rem;">Despite these measures, risks remain including but not limited to: prompt injection, token exhaustion, data leakage, sandbox escapes, unintended file modifications, and unpredictable agent behavior.</p>
</div>

<form method="GET" action="/app">
<button type="submit" style="width:100%;padding:14px;background:#2563eb;color:white;border:none;border-radius:10px;font-size:1rem;font-weight:bold;cursor:pointer;transition:background 0.2s;">
I have read, understood, and agreed to the LICENSE and Security Notice
</button>
</form>
<p style="text-align:center;margin-top:1rem;color:#64748b;font-size:0.85rem;">You must accept this notice to enter the application.</p>
</div>
"""

# ── Approval System ──────────────────────────────────────────────────────────

class APIApprovalSystem:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = False
        self._tool_name = ""
        self._tool_args = {}
        self._result = False
        self._event = threading.Event()
        self.websockets: list[WebSocket] = []

    def request_approval(self, tool_name: str, args: dict) -> bool:
        with self._lock:
            self._pending = True
            self._tool_name = tool_name
            self._tool_args = args
            self._event.clear()

        if loop is not None:
            safe_args = {}
            for k, v in args.items():
                s = str(v)
                safe_args[k] = s[:500] if len(s) > 500 else s
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "approval_request", "tool": tool_name, "args": safe_args}),
                loop
            )

        self._event.wait(timeout=300)
        with self._lock:
            self._pending = False
            return self._result

    def check_pending(self) -> bool:
        with self._lock:
            return self._pending

    def respond(self, approved: bool):
        with self._lock:
            self._result = approved
            self._pending = False
        self._event.set()

    async def broadcast(self, msg):
        for ws in list(self.websockets):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(msg)
            except Exception:
                pass

approval_system = APIApprovalSystem()

def approval_callback(tool_name: str, args: dict) -> bool:
    if agent.emergency_stop:
        return False
    return approval_system.request_approval(tool_name, args)

agent.approval_callback = approval_callback


# ── HUMAN_LLM System ─────────────────────────────────────────────────────────

class HumanLLMBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._response: dict | None = None

    def request_tool(self, iteration: int) -> dict | None:
        with self._lock:
            self._response = None
            self._event.clear()

        if loop is not None:
            asyncio.run_coroutine_threadsafe(
                approval_system.broadcast({"type": "human_llm_request", "iteration": iteration}),
                loop
            )

        self._event.wait(timeout=600)
        with self._lock:
            return self._response

    def respond(self, data: dict):
        with self._lock:
            self._response = data
        self._event.set()

human_llm = HumanLLMBridge()

if os.getenv("HUMAN_LLM", "false").lower() == "true":
    agent.human_llm_callback = human_llm.request_tool


# ── Status Callback ──────────────────────────────────────────────────────────

def status_callback(data: dict):
    if loop is not None:
        asyncio.run_coroutine_threadsafe(approval_system.broadcast(data), loop)

agent.status_callback = status_callback
agent.step_logger.log_callback = lambda d: status_callback(d) if agent.verbose else None


# ── Chat State (for feedback) ────────────────────────────────────────────────
# We track the last prompt so feedback can reference the correct prompt/response pair
_chat_state: dict[str, dict] = {}  # websocket_id -> {"last_prompt": str, "last_response": str}


# ── Chat Command Handlers ────────────────────────────────────────────────────

HELP_TEXT = """### 🐺 GrimmBot Command Center
*Use these commands to control the agent's behavior and access system features.*

---
#### ⚙️ System Controls
* **`!emergency`** — Instantly halt all agent operations.
* **`!verbose`** — Toggle detailed thought and tool execution logs.
* **`!throttle [N/off]`** — Set a delay (in seconds) between agent actions.
* **`!commssafeguard`** — Toggle approval requirements for text entry/typing.

#### 🌐 Browser & Session
* **`!login <url> [profile]`** — Open a browser tab for manual VNC login.
* **`!profiles`** — List available Chromium profiles.
* **`!wipe <profile>`** — Erase a specific browser profile (or `ALL`).

#### 🗄️ Memory & Data
* **`!files`** — List all files stored in the output wormhole.
* **`!memory [show/reset]`** — View or clear the agent's persistent memory for the current profile.
* **`!schedule`** — View all background scheduled tasks.

#### 🛠️ Tools
* **`!custom-tools`** — List all dynamically generated Python tools.
* **`!tools`** — List all default built-in capabilities.
* **`!help`** — Show this command menu.
---
"""

async def handle_command(prompt: str, profile: str, websocket: WebSocket) -> bool:
    lower = prompt.lower().strip()

    if not lower.startswith("!"):
        return False

    parts = lower.split(maxsplit=2)
    cmd = parts[0]

    try:
        if cmd == "!help":
            await websocket.send_json({"type": "result", "answer": HELP_TEXT, "steps": 0})
        elif cmd == "!emergency":
            agent.emergency_stop = True
            if approval_system.check_pending():
                approval_system.respond(False)
            await websocket.send_json({"type": "result", "answer": "🛑 EMERGENCY STOP initiated.", "steps": 0})
        elif cmd == "!verbose":
            agent.verbose = not agent.verbose
            agent.save_settings()
            state = "ON" if agent.verbose else "OFF"
            await websocket.send_json({"type": "result", "answer": f"🔍 Verbose mode: **{state}**\n(Tool use, thoughts, and intermediate steps will be shown)", "steps": 0})
        elif cmd == "!reset":
            reset_memory(profile)
            await websocket.send_json({"type": "result", "answer": f"🧹 Memory reset for profile '{profile}'.", "steps": 0})
        elif cmd == "!throttle":
            if len(parts) < 2:
                await websocket.send_json({"type": "result", "answer": f"⏱️ Throttle is currently: {agent.throttle_seconds}s", "steps": 0})
            else:
                val = parts[1]
                if val == "off":
                    agent.throttle_seconds = 0
                    agent.save_settings()
                    await websocket.send_json({"type": "result", "answer": "⏱️ Throttle disabled.", "steps": 0})
                elif val == "on":
                    agent.throttle_seconds = 2
                    agent.save_settings()
                    await websocket.send_json({"type": "result", "answer": "⏱️ Throttle set to 2s.", "steps": 0})
                else:
                    try:
                        n = int(val)
                        agent.throttle_seconds = max(0, n)
                        agent.save_settings()
                        await websocket.send_json({"type": "result", "answer": f"⏱️ Throttle set to {agent.throttle_seconds}s.", "steps": 0})
                    except ValueError:
                        await websocket.send_json({"type": "result", "answer": "Usage: `!throttle [N/on/off]`", "steps": 0})
        elif cmd == "!commssafeguard":
            agent.commssafeguard = not agent.commssafeguard
            agent.save_settings()
            state = "ON" if agent.commssafeguard else "OFF"
            await websocket.send_json({"type": "result", "answer": f"🛡️ Communication Safeguard mode: **{state}**", "steps": 0})
        elif cmd == "!login":
            url = parts[1] if len(parts) > 1 else ""
            login_profile = parts[2] if len(parts) > 2 else "Default"
            if not url:
                await websocket.send_json({"type": "result", "answer": "Usage: `!login <url> [profile]`", "steps": 0})
            else:
                if not is_chromium_running():
                    launch_chromium(url, login_profile)
                    await websocket.send_json({"type": "result", "answer": f"🔑 Browser opened to `{url}` with profile `{login_profile}`. Use the VNC panel to log in manually, then send any message when done.", "steps": 0})
                else:
                    from screen import chromium_navigate as screen_navigate
                    screen_navigate(url)
                    await websocket.send_json({"type": "result", "answer": f"🔑 Navigated to `{url}`. Use the VNC panel to log in manually.", "steps": 0})
        elif cmd == "!profiles":
            result = list_chromium_profiles()
            await websocket.send_json({"type": "result", "answer": f"🗂️ {result}", "steps": 0})
        elif cmd == "!wipe":
            if len(parts) < 2:
                await websocket.send_json({"type": "result", "answer": "Usage: `!wipe <profile>` or `!wipe ALL`", "steps": 0})
            else:
                result = wipe_chromium_profile(parts[1])
                await websocket.send_json({"type": "result", "answer": f"🗑️ {result}", "steps": 0})
        elif cmd == "!files":
            wormhole = Path(config.wormhole_dir)
            if not wormhole.exists():
                await websocket.send_json({"type": "result", "answer": "Wormhole directory not found.", "steps": 0})
            else:
                files = []
                for f in sorted(wormhole.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(wormhole)
                        size = f.stat().st_size
                        files.append(f"  `{rel}` ({size:,} bytes)")
                if files:
                    listing = "\n".join(files[:50])
                    await websocket.send_json({"type": "result", "answer": f"📁 **Wormhole files:**\n{listing}", "steps": 0})
                else:
                    await websocket.send_json({"type": "result", "answer": "📁 Wormhole is empty.", "steps": 0})
        elif cmd == "!memory":
            sub = parts[1] if len(parts) > 1 else "show"
            mem = get_memory(profile)
            if sub == "reset":
                reset_memory(profile)
                await websocket.send_json({"type": "result", "answer": f"🧠 Memory wiped for profile '{profile}'.", "steps": 0})
            else:
                entries = mem.list_entries() if hasattr(mem, 'list_entries') else []
                if entries:
                    lines = [f"  [{e.timestamp}] {e.result_summary[:80]}" for e in entries[:20]]
                    await websocket.send_json({"type": "result", "answer": f"🧠 **Memory ({len(entries)} entries):**\n" + "\n".join(lines), "steps": 0})
                else:
                    await websocket.send_json({"type": "result", "answer": f"🧠 No memories stored for profile '{profile}'.", "steps": 0})
        elif cmd == "!schedule":
            scheduler = get_scheduler()
            tasks = scheduler.list_tasks()
            if tasks:
                lines = [f"  {'[ON]' if t.enabled else '[OFF]'} `{t.id}`: {t.prompt[:60]}" for t in tasks]
                await websocket.send_json({"type": "result", "answer": f"📅 **Scheduled tasks:**\n" + "\n".join(lines), "steps": 0})
            else:
                await websocket.send_json({"type": "result", "answer": "📅 No scheduled tasks.", "steps": 0})
        elif cmd == "!custom-tools":
            tl = agent.custom_tools.list_tools()
            if tl:
                await websocket.send_json({"type": "result", "answer": f"🛠️ **Custom tools:** {', '.join(tl)}", "steps": 0})
            else:
                await websocket.send_json({"type": "result", "answer": "🛠️ No custom tools registered.", "steps": 0})
        elif cmd == "!tools":
            names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
            formatted = ", ".join(f"`{n}`" for n in names)
            await websocket.send_json({"type": "result", "answer": f"🔧 **Built-in tools ({len(names)}):**\n{formatted}", "steps": 0})
        else:
            await websocket.send_json({"type": "result", "answer": f"Unknown command: `{cmd}`. Type `!help` for available commands.", "steps": 0})
    except Exception as e:
        logger.error("Command error: %s", e)
        await websocket.send_json({"type": "error", "msg": f"Command failed: {type(e).__name__}"})

    return True


# ── FastAPI App ───────────────────────────────────────────────────────────────

loop = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop
    loop = asyncio.get_running_loop()

    scheduler = get_scheduler()

    def on_scheduled(task: ScheduledTask):
        if agent.emergency_stop:
            return
        try:
            result = agent.run_task(task.prompt, task.profile)
            logger.info("Scheduled: %s -> %s", task.id, result.answer[:100])
        except Exception as e:
            logger.error("Scheduled failed: %s -> %s", task.id, e)

    scheduler.add_callback(on_scheduled)
    scheduler.start()

    yield
    if is_chromium_running():
        close_chromium()

app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

if os.path.exists("/usr/share/novnc"):
    app.mount("/novnc", StaticFiles(directory="/usr/share/novnc"), name="novnc")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(
        content=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>GrimmBot - Welcome</title></head>
<body style="background:#0f111a;min-height:100vh;display:flex;align-items:center;justify-content:center;">
{DISCLAIMER_TEXT}
</body></html>""",
        status_code=200,
    )


@app.get("/app", response_class=HTMLResponse)
async def get_app():
    html_path = Path("assets/index.html")
    if not html_path.exists():
         return HTMLResponse("Error: Application assets not found.", status_code=404)
    content = html_path.read_text()
    if not is_local_llm:
        content = content.replace('<button class="btn-export" id="btn-export-rlhf" title="Export feedback as RLHF dataset">📊 Export RLHF</button>', '<button style="display:none;" id="btn-export-rlhf"></button>')
        content = content.replace("if (sender === 'agent' && !isError) {", "if (false) {")
    return HTMLResponse(content=content, status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    approval_system.websockets.append(websocket)
    ws_id = id(websocket)
    _chat_state[ws_id] = {"last_prompt": "", "last_response": ""}
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "msg": "Invalid message format."})
                continue

            if msg.get("type") == "chat":
                prompt = msg.get("text", "").strip()
                if not prompt:
                    continue
                prompt = prompt[:10000]
                profile = msg.get("profile", "default")

                was_command = await handle_command(prompt, profile, websocket)
                if was_command:
                    continue

                _chat_state[ws_id]["last_prompt"] = prompt
                threading.Thread(target=run_agent_task, args=(prompt, profile, websocket, ws_id), daemon=True).start()

            elif msg.get("type") == "approval_response":
                if "require_future" in msg and msg.get("tool"):
                    agent.custom_tools.set_approval_requirement(msg.get("tool"), msg.get("require_future"))
                approval_system.respond(msg.get("approved", False))

            elif msg.get("type") == "emergency_stop":
                agent.emergency_stop = True
                if approval_system.check_pending():
                    approval_system.respond(False)

            elif msg.get("type") == "human_llm_response":
                human_llm.respond({"tool": msg.get("tool", "done"), "args": msg.get("args", "{}")})

            elif msg.get("type") == "human_llm_cancel":
                human_llm.respond(None)

            elif msg.get("type") == "feedback":
                if not is_local_llm:
                    continue
                # Store user feedback for RLHF dataset
                feedback = get_feedback_store(config.feedback_file)
                prompt_text = msg.get("prompt", _chat_state.get(ws_id, {}).get("last_prompt", ""))
                response_text = msg.get("response", _chat_state.get(ws_id, {}).get("last_response", ""))
                rating = msg.get("rating", "positive")
                feedback.add(prompt_text, response_text, rating, msg.get("message_id", ""))
                await websocket.send_json({"type": "feedback_ack", "rating": rating})

            elif msg.get("type") == "export_rlhf":
                if not is_local_llm:
                    continue
                # Export RLHF dataset
                feedback = get_feedback_store(config.feedback_file)
                result = feedback.export_jsonl(str(Path(config.data_dir) / "rlhf_dataset.jsonl"))
                await websocket.send_json({"type": "result", "answer": f"📊 {result}", "steps": 0})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", type(e).__name__)
    finally:
        if websocket in approval_system.websockets:
            approval_system.websockets.remove(websocket)
        _chat_state.pop(ws_id, None)


@app.websocket("/websockify")
async def vnc_proxy(websocket: WebSocket):
    import asyncio
    import websockets as ws_client

    await websocket.accept()

    try:
        async with ws_client.connect("ws://localhost:6080") as target:
            async def forward_to_client():
                async for msg in target:
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_text(msg)

            async def forward_to_vnc():
                while True:
                    msg = await websocket.receive_bytes()
                    await target.send(msg)

            await asyncio.gather(forward_to_client(), forward_to_vnc())
    except Exception as e:
         pass
    finally:
        try: await websocket.close()
        except: pass


def run_agent_task(prompt: str, profile: str, websocket: WebSocket, ws_id: int = 0):
    agent.emergency_stop = False
    try:
        if loop:
            asyncio.run_coroutine_threadsafe(websocket.send_json({"type": "status", "msg": "working"}), loop)

        result = agent.run_task(prompt, profile)

        # Store for feedback reference
        if ws_id in _chat_state:
            _chat_state[ws_id]["last_response"] = result.answer

        if loop and not agent.emergency_stop:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "result", "answer": result.answer, "steps": result.steps}), loop)
        elif loop and agent.emergency_stop:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "error", "msg": "Task halted by emergency stop."}), loop)

    except Exception as e:
        safe_msg = f"An internal error occurred ({type(e).__name__}). Check server logs."
        logger.error("Agent task error: %s", e)
        if loop:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "error", "msg": safe_msg}), loop)
    finally:
        if loop:
            asyncio.run_coroutine_threadsafe(websocket.send_json({"type": "status", "msg": "idle"}), loop)


if __name__ == "__main__":
    uvicorn.run("grimmbot:app", host="0.0.0.0", port=5000, log_level="info")