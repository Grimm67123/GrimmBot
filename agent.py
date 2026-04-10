"""
GrimmBot — Autonomous Agent
Full computer control (inside Docker) via vision + DOM.
"""

import os
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_DEBUG"] = "True"
import litellm
litellm.suppress_debug_info = True

import re
import io
import json
import time
import random
import shlex
import hashlib
import logging
import base64
import subprocess
import shutil
import difflib
import traceback
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime

from litellm import completion

from core import (
    AgentConfig, CustomToolRegistry, TOOL_DEFINITIONS,
    SYSTEM_PROMPT_VISION, SYSTEM_PROMPT_TEXT,
    init_safe_paths, is_domain_allowed, is_command_allowed,
)
from memory import get_memory, MemoryConfig, retrieve_relevant_rules
from screen import (
    take_screenshot_raw, screenshot_to_base64,
    save_screenshot, mouse_move, mouse_click, mouse_double_click,
    mouse_scroll, mouse_drag, keyboard_type, keyboard_press,
    keyboard_shortcut, clipboard_copy, clipboard_paste, clipboard_get,
    clipboard_set, launch_chromium, close_chromium, chromium_navigate,
    chromium_new_tab, chromium_close_tab, chromium_switch_tab,
    chromium_refresh, chromium_back, chromium_forward, is_chromium_running,
    list_chromium_profiles, wipe_chromium_profile, get_active_window,
    focus_window, wait_for_screen_change, wait_for_screen_stable,
    read_true_dom, SCREEN_WIDTH, SCREEN_HEIGHT,
)
from tools import Tools

logger = logging.getLogger("agent")


# ── Task Result & Logger ─────────────────────────────────────────────────────


@dataclass
class TaskResult:
    answer: str
    steps: int
    screenshot: Optional[bytes] = None
    output_files: list[str] = field(default_factory=list)


class StepLogger:
    ICONS = {
        "screenshot": "📸", "click": "🖱️", "double_click": "🖱️",
        "type_text": "⌨️", "press_key": "⌨️", "hotkey": "⌨️", "scroll": "📜",
        "open_browser": "🌐", "close_browser": "🌐", "go_to_url": "🔗",
        "new_tab": "📑", "close_tab": "📑", "shell": "💻",
        "read_file": "📖", "read_file_lines": "📖", "write_file": "✏️",
        "patch_file": "🔧", "insert_at_line": "📝", "delete_lines": "🗑️",
        "find_in_files": "🔍", "read_page_text": "📄", "read_page_source": "📄",
        "monitor_page_text": "👁️", "monitor_page_change": "👁️",
        "monitor_page_element_count": "👁️", "monitor_pixel_region": "👁️",
        "monitor_multi_condition": "👁️", "wait_for_pixel_color": "👁️",
        "remember": "🧠", "recall": "🧠", "schedule_task": "📅",
        "create_plan": "📋", "update_plan_step": "📋",
        "create_custom_tool": "🛠️", "list_custom_tools": "🛠️",
        "save_adaptation_rule": "📝",
        "wait": "⏳", "wait_for_change": "⏳", "wait_for_stable": "⏳",
        "done": "✅", "copy": "📋", "paste": "📋",
        "list_directory": "📂", "delete_file": "🗑️",
    }

    def __init__(self):
        self.debug_mode = False
        self.log_callback: Optional[Callable] = None

    def _broadcast(self, text: str):
        if self.log_callback:
            self.log_callback({"type": "verbose_log", "msg": text})

    def log_step(self, step, tool, args, result):
        icon = self.ICONS.get(tool, "🔧")
        a = self._fmt_args(tool, args)
        print(f"  {icon} Step {step}: {tool}({a})")
        r = self._fmt_result(result)
        if r:
            for line in r.splitlines()[:3]:
                print(f"     -> {line}")
            if len(r.splitlines()) > 3:
                print(f"     -> ...({len(r.splitlines())-3} more)")

        if tool != "screenshot":
            md = f"**{icon} tool:** `{tool}`"
            if args:
                if len(args) > 1 and "code" in args:
                    md += f"\n*Args:*\n```python\n{args.get('code')}\n```"
                else:
                    md += f"\n*Args:* `{a}`"
            if r:
                md += f"\n*Result:*\n```\n{r[:1500]}\n```"
            self._broadcast(md)

    def log_thinking(self, content):
        if content and content.strip():
            print(f"  💭 {content.strip()[:200]}")
            self._broadcast(f"💭 **Thought Process:**\n> {content.strip()}")

    def log_error(self, error):
        print(f"  ⚠️  {error}")
        self._broadcast(f"⚠️ **Agent Error:**\n`{error}`")

    def log_api_call(self, model, count):
        if self.debug_mode:
            print(f"  📡 -> {model} ({count} msgs)")

    def log_api_response(self, resp):
        if self.debug_mode:
            try:
                u = resp.usage
                if u:
                    print(f"  📡 <- {u.prompt_tokens}->{u.completion_tokens} tokens")
            except Exception:
                pass

    def _fmt_args(self, tool, args):
        if not args:
            return ""
        if tool == "click":
            return f"{args.get('x')}, {args.get('y')}"
        if tool == "type_text":
            t = args.get("text", "")
            return f'"{t[:40]}{"..." if len(t)>40 else ""}"'
        if tool in ("go_to_url",):
            return args.get("url", "")[:60]
        if tool == "shell":
            return args.get("command", "")[:60]
        if tool in ("read_file", "write_file", "patch_file", "save_adaptation_rule"):
            return args.get("path", args.get("rule", ""))[:60]
        if tool == "monitor_page_text":
            return f'"{args.get("watch_for","")[:30]}"'
        if tool == "done":
            return args.get("result", "")[:50]
        if tool == "create_custom_tool":
            return args.get("name", "")
        parts = [f"{k}={str(v)[:30]}" for k, v in list(args.items())[:3]]
        return ", ".join(parts)

    def _fmt_result(self, r):
        if not r or r.startswith("Screenshot captured"):
            return ""
        return r[:300] + "..." if len(r) > 300 else r


# ── Agent ─────────────────────────────────────────────────────────────────────


class GrimmAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig.from_env()
        self.memory_config = MemoryConfig.from_env()
        self.throttle_seconds = 0
        self.commssafeguard = True
        self.verbose = False
        self.emergency_stop = False
        self.approval_callback: Optional[Callable] = None
        self.human_llm_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.step_logger = StepLogger()
        self._last_screenshot_hash: Optional[str] = None
        self.custom_tools = CustomToolRegistry(self.config.custom_tools_dir)
        self._settings_file = Path(self.config.settings_file)
        self._load_settings()

        init_safe_paths(self.config)
        for d in [self.config.wormhole_dir, self.config.workspace_dir,
                   self.config.profile_dir, self.config.custom_tools_dir,
                   self.config.data_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _load_settings(self):
        if self._settings_file.exists():
            try:
                data = json.loads(self._settings_file.read_text())
                self.throttle_seconds = data.get("throttle_seconds", 0)
                self.commssafeguard = data.get("commssafeguard", True)
                self.verbose = data.get("verbose", False)
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")

    def save_settings(self):
        try:
            self._settings_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "throttle_seconds": self.throttle_seconds,
                "commssafeguard": self.commssafeguard,
                "verbose": self.verbose
            }
            self._settings_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def _get_tool_defs(self):
        defs = list(TOOL_DEFINITIONS)
        if not self.config.use_vision:
            skip = {"screenshot", "wait_for_pixel_color", "monitor_pixel_region"}
            defs = [t for t in defs if t["function"]["name"] not in skip]
        defs.extend(self.custom_tools.get_definitions())
        return defs

    def _check_approval(self, func, args):
        need = False

        if func in ["shell", "write_file", "delete_file", "wipe_profile", "patch_file", "insert_at_line", "delete_lines", "create_custom_tool"]:
            need = True

        elif func in self.custom_tools._functions:
            need = self.custom_tools._requires_approval.get(func, True)

        elif func == "go_to_url" and not is_domain_allowed(args.get("url", ""), self.config):
            need = True
            reason = f"Domain '{args.get('url')}' is not in the allowed list"

        elif self.commssafeguard:
            if func == "type_text" or (func == "press_key" and args.get("key", "").lower() == "enter"):
                need = True
                reason = "Agent is about to send input or press Enter (commssafeguard is ON)"
            elif func == "click_element":
                eid = str(args.get("element_id", ""))
                from screen import INTERACTABLE_MAP
                meta = INTERACTABLE_MAP.get(eid, {})
                label = meta.get("label", "").lower()
                comms_keywords = ["send", "submit", "reply", "post", "chat", "message"]
                if any(k in label for k in comms_keywords):
                    need = True
                    reason = f"Agent is clicking a potential communication/submit button: '{label}'"
            elif func == "click":
                need = True
                reason = "Agent is performing a raw click action while commssafeguard is ON"

        if need and self.approval_callback:
            return self.approval_callback(func, args)
        return True

    def _build_completion_kwargs(self, model, messages, tool_defs):
        """Build the kwargs dict for litellm.completion(), handling local models."""
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": tool_defs,
            "tool_choice": "auto",
            "timeout": self.config.api_timeout,
        }

        # Pass API base for local models (Ollama, LM Studio, etc.)
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        # Pass API key if set
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        return kwargs

    def run_task(self, user_prompt: str, profile: str = "default") -> TaskResult:
        start_time = time.time()
        tools = Tools(self.config)
        tools.current_profile = profile
        memory = get_memory(profile)
        mem_ctx = ""
        if self.memory_config.enabled:
            ctx = memory.get_context(user_prompt)
            if ctx:
                mem_ctx = f"\n=== YOUR MEMORY ===\n{ctx}\n===================\n"

        # RAG-based adaptation: only load rules relevant to the current prompt
        adaptation_ctx = ""
        try:
            adap_path = Path(self.config.adaptation_file)
            if adap_path.exists():
                all_rules = json.loads(adap_path.read_text())
                relevant_rules = retrieve_relevant_rules(all_rules, user_prompt)
                if relevant_rules:
                    adaptation_ctx = "\nCRITICAL - LEARNED RULES (relevant to this task):\n"
                    for i, rule in enumerate(relevant_rules, 1):
                        adaptation_ctx += f"{i}. {rule}\n"
        except Exception as e:
            logger.error(f"Failed to load adaptations: {e}")

        system = (SYSTEM_PROMPT_VISION if self.config.use_vision else SYSTEM_PROMPT_TEXT).format(
            memory_context=mem_ctx,
            adaptation_context=adaptation_ctx,
            max_iterations=self.config.max_iterations,
        )

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"<USER_TASK>\n{user_prompt}\n</USER_TASK>"}]
        tool_defs = self._get_tool_defs()
        steps = 0
        last_ss = None
        self._last_screenshot_hash = None
        model = self.config.model

        try:
            for _ in range(self.config.max_iterations):
                if self.emergency_stop:
                    return TaskResult("Emergency stop", steps, last_ss)
                if time.time() - start_time > self.config.task_timeout:
                    return TaskResult("Task timed out", steps, last_ss)

                self.step_logger.log_api_call(model, len(messages))
                # Human LLM mode: request tool call from the user via callback
                if os.getenv("HUMAN_LLM", "false").lower() == "true" and self.human_llm_callback:
                    human_input = self.human_llm_callback(_ + 1)
                    if human_input is None:
                        return TaskResult("Human aborted.", steps, last_ss)

                    tool_name = human_input.get("tool", "done")
                    args_str = human_input.get("args", "{}")
                    if tool_name.lower() == "done":
                        args_str = json.dumps({"result": args_str if args_str != "{}" else "Done"})

                    from types import SimpleNamespace
                    import uuid
                    tc_id = f"call_{uuid.uuid4().hex[:8]}"
                    tc = SimpleNamespace(id=tc_id, function=SimpleNamespace(name=tool_name, arguments=args_str))
                    _tc_id = tc_id; _tool_name = tool_name; _args_str = args_str
                    msg = SimpleNamespace(
                        content="Human override active.",
                        tool_calls=[tc],
                        model_dump=lambda: {
                            "role": "assistant", "content": "Human override active.",
                            "tool_calls": [{"id": _tc_id, "type": "function", "function": {"name": _tool_name, "arguments": _args_str}}]
                        }
                    )
                    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])

                else:
                    try:
                        local_api_prefixes = ["ollama", "lm_studio", "localai", "vllm", "text-gen"]
                        is_local = self.config.api_base and any(model.startswith(p) for p in local_api_prefixes)
                        if is_local:
                            import requests as req_lib
                            from types import SimpleNamespace
                            import uuid

                            m_name = model.split("/", 1)[-1] if "/" in model else model
                            raw_base = self.config.api_base.rstrip("/")
                            use_ollama_native = model.startswith("ollama")

                            headers = {"Content-Type": "application/json"}
                            if self.config.api_key:
                                headers["Authorization"] = f"Bearer {self.config.api_key}"

                            if use_ollama_native:
                                # Ollama native /api/chat — bypasses OpenAI compat layer
                                base = raw_base
                                if base.endswith("/v1"):
                                    base = base[:-3].rstrip("/")
                                api_url = f"{base}/api/chat"
                            else:
                                api_base = raw_base
                                if not api_base.endswith("/v1"):
                                    api_base += "/v1"
                                api_url = f"{api_base}/chat/completions"

                            payload = {
                                "model": m_name,
                                "messages": messages,
                                "stream": True,
                            }
                            if tool_defs:
                                payload["tools"] = tool_defs

                            try:
                                r = req_lib.post(
                                    api_url, json=payload, headers=headers,
                                    stream=True, timeout=(30, self.config.api_timeout),
                                )
                                r.raise_for_status()
                            except req_lib.exceptions.Timeout:
                                raise Exception("API timeout")
                            except req_lib.exceptions.HTTPError as he:
                                body = ""
                                try: body = r.text[:500]
                                except: pass
                                raise Exception(f"Local API error: {he}. {body}")
                            except req_lib.exceptions.RequestException as re_exc:
                                raise Exception(f"Local API connection error: {re_exc}")

                            # --- Accumulate streamed response ---
                            full_content = ""
                            raw_tool_calls = []

                            try:
                                if use_ollama_native:
                                    # Ollama native: newline-delimited JSON
                                    for line in r.iter_lines(decode_unicode=True):
                                        if not line:
                                            continue
                                        try:
                                            chunk = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue
                                        mp = chunk.get("message", {})
                                        if mp.get("content"):
                                            full_content += mp["content"]
                                        if chunk.get("done"):
                                            raw_tool_calls = mp.get("tool_calls", [])
                                            break
                                else:
                                    # OpenAI-compatible SSE streaming
                                    _tc_accum = {}
                                    for line in r.iter_lines(decode_unicode=True):
                                        if not line or not line.startswith("data: "):
                                            continue
                                        data_str = line[6:]
                                        if data_str.strip() == "[DONE]":
                                            break
                                        try:
                                            chunk = json.loads(data_str)
                                        except json.JSONDecodeError:
                                            continue
                                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                                        if delta.get("content"):
                                            full_content += delta["content"]
                                        for tcd in delta.get("tool_calls", []):
                                            idx = tcd.get("index", 0)
                                            if idx not in _tc_accum:
                                                _tc_accum[idx] = {"id": tcd.get("id", f"call_{uuid.uuid4().hex[:8]}"), "function": {"name": "", "arguments": ""}}
                                            fn = tcd.get("function", {})
                                            if "name" in fn:
                                                _tc_accum[idx]["function"]["name"] += fn["name"]
                                            if "arguments" in fn:
                                                _tc_accum[idx]["function"]["arguments"] += fn["arguments"]
                                    raw_tool_calls = [_tc_accum[i] for i in sorted(_tc_accum.keys())]
                            finally:
                                r.close()

                            # --- Build response objects ---
                            tc_list = []
                            for tc in raw_tool_calls:
                                fn_dict = tc.get("function", {})
                                args_val = fn_dict.get("arguments", {})
                                args_str = json.dumps(args_val) if isinstance(args_val, dict) else (args_val or "{}")
                                tc_list.append(SimpleNamespace(
                                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                    function=SimpleNamespace(name=fn_dict.get("name", ""), arguments=args_str)
                                ))

                            _dump_dict = {
                                "role": "assistant",
                                "content": full_content,
                            }
                            if tc_list:
                                _dump_dict["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tc_list]

                            msg = SimpleNamespace(
                                content=full_content,
                                tool_calls=tc_list if tc_list else None,
                                model_dump=lambda d=_dump_dict: d
                            )
                            resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
                        else:
                            comp_kwargs = self._build_completion_kwargs(model, messages, tool_defs)
                            resp = completion(**comp_kwargs)
                    except Exception as e:
                        em = str(e)
                        if "rate_limit" in em.lower():
                            self.step_logger.log_error("Rate limited -- waiting 30s")
                            time.sleep(30)
                            continue
                        elif "timeout" in em.lower():
                            self.step_logger.log_error("API timeout -- retrying in 5s. (Check that your local model server is running and reachable)")
                            time.sleep(5)
                            continue
                        self.step_logger.log_error(f"API error: {em[:200]}")
                        return TaskResult(f"LLM error: {em[:200]}", steps, last_ss)

                self.step_logger.log_api_response(resp)
                if not resp.choices:
                    return TaskResult("Empty LLM response", steps, last_ss)

                msg = resp.choices[0].message
                if not msg.tool_calls:
                    answer = msg.content or "Task completed."
                    self.step_logger.log_thinking(msg.content or "")
                    if self.memory_config.enabled:
                        memory.add(task=user_prompt[:200], result=answer[:500], tags=["task"])
                    return TaskResult(answer, steps, last_ss)

                if msg.content:
                    self.step_logger.log_thinking(msg.content)
                messages.append(msg.model_dump())

                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    if fn == "screenshot":
                        if not self.config.use_vision:
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "No screenshots in text mode. Use read_page_text()."})
                            steps += 1
                            self.step_logger.log_step(steps, fn, args, "N/A")
                            continue
                        b64 = screenshot_to_base64()
                        if b64:
                            h = hashlib.md5(b64[:2000].encode()).hexdigest()
                            if h == self._last_screenshot_hash:
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Screen unchanged. No new image sent."})
                            else:
                                self._last_screenshot_hash = h
                                last_ss = take_screenshot_raw()
                                messages.append({"role": "tool", "tool_call_id": tc.id, "content": [
                                    {"type": "text", "text": "Screenshot captured. Grid: 100px lines, 50px ticks. 1920x1080."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                ]})
                                if self.config.vision_model != self.config.model:
                                    model = self.config.vision_model
                        else:
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Screenshot failed."})
                        steps += 1
                        self.step_logger.log_step(steps, fn, args, "captured")
                        continue

                    if fn == "done":
                        rt = args.get("result", "Done")
                        if self.memory_config.enabled:
                            memory.add(task=user_prompt[:200], result=rt[:500], tags=["task"])
                        steps += 1
                        self.step_logger.log_step(steps, fn, args, rt[:100])
                        return TaskResult(rt, steps, last_ss)

                    if self.emergency_stop:
                        return TaskResult("Emergency stop", steps, last_ss)
                    if not self._check_approval(fn, args):
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ACTION_DENIED"})
                        steps += 1
                        self.step_logger.log_step(steps, fn, args, "DENIED")
                        continue

                    # Custom tool routing
                    if fn == "create_custom_tool":
                        result = self.custom_tools.create_tool(
                            args.get("name", ""), args.get("description", ""),
                            args.get("parameters", {"type": "object", "properties": {}}),
                            args.get("code", ""),
                            args.get("requires_approval", True))
                        tool_defs = self._get_tool_defs()
                    elif fn == "list_custom_tools":
                        tl = self.custom_tools.list_tools()
                        result = f"Custom tools: {', '.join(tl)}" if tl else "No custom tools."
                    elif fn == "delete_custom_tool":
                        result = self.custom_tools.delete_tool(args.get("name", ""))
                        tool_defs = self._get_tool_defs()
                    elif hasattr(tools, fn):
                        try:
                            result = getattr(tools, fn)(**args)
                        except TypeError as e:
                            result = f"Invalid arguments: {e}"
                        except Exception as e:
                            result = f"Tool error: {e}"
                    elif fn in self.custom_tools._functions:
                        result = self.custom_tools.call(fn, args)
                    else:
                        result = f"Unknown tool: {fn}"

                    if fn in ("click", "double_click", "type_text", "press_key", "hotkey",
                              "scroll", "open_browser", "go_to_url", "new_tab", "close_tab",
                              "switch_tab", "refresh_page", "go_back", "go_forward", "paste", "drag"):
                        self._last_screenshot_hash = None
                        model = self.config.model

                    steps += 1
                    self.step_logger.log_step(steps, fn, args, str(result))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)[:6000]})

                    if self.throttle_seconds > 0:
                        remaining = self.throttle_seconds
                        while remaining > 0:
                            if self.status_callback:
                                self.status_callback({"type": "throttle", "remaining": remaining, "total": self.throttle_seconds})
                            time.sleep(1)
                            remaining -= 1
                        if self.status_callback:
                            self.status_callback({"type": "throttle", "remaining": 0, "total": self.throttle_seconds})

            return TaskResult("Max iterations reached", steps, last_ss)
        finally:
            if last_ss:
                try:
                    Path(self.config.wormhole_dir, "last_screenshot.png").write_bytes(last_ss)
                except Exception:
                    pass