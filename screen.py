"""
Screen interaction engine for GrimmBot.

True DOM extraction via CDP, raw screenshots, Chromium management, 
and native OS mouse/keyboard automation.
"""

import os
import io
import time
import base64
import shutil
import random
import logging
import hashlib
import urllib.request
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pyautogui
import websocket  # pip install websocket-client

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.03

logger = logging.getLogger("screen")

DISPLAY = os.getenv("DISPLAY", ":99")
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CHROMIUM_PROFILE_DIR = os.getenv("CHROMIUM_PROFILE_DIR", "/home/grimmbot/.config/chromium")

# Global mapping to store interactive element coordinates from the DOM
INTERACTABLE_MAP = {}


# ── Raw Screenshots & Hashing ────────────────────────────────────────────────

def take_screenshot_raw() -> Optional[bytes]:
    """Capture raw screenshot via scrot. Used strictly for debugging or human oversight."""
    try:
        result = subprocess.run(
            ["scrot", "-o", "/tmp/screenshot.png"],
            capture_output=True, timeout=10,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        if result.returncode != 0:
            logger.error("scrot failed: %s", result.stderr.decode())
            return None
        with open("/tmp/screenshot.png", "rb") as f:
            return f.read()
    except Exception as e:
        logger.error("Screenshot capture failed: %s", e)
        return None

def screenshot_to_base64() -> Optional[str]:
    """Return raw screenshot as base64-encoded PNG string."""
    data = take_screenshot_raw()
    if data:
        return base64.b64encode(data).decode("ascii")
    return None

def save_screenshot(path: str) -> bool:
    """Save raw screenshot to disk."""
    data = take_screenshot_raw()
    if data:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        return True
    return False

def _screenshot_hash() -> Optional[str]:
    """Fast hash of the raw screen bytes for visual change detection."""
    raw = take_screenshot_raw()
    if not raw:
        return None
    return hashlib.md5(raw).hexdigest()

def wait_for_screen_change(timeout: float = 10.0, poll_interval: float = 0.5) -> str:
    start = time.time()
    initial_hash = _screenshot_hash()
    if not initial_hash:
        time.sleep(min(timeout, 2.0))
        return "Waited (could not capture screen)"
    while time.time() - start < timeout:
        time.sleep(poll_interval)
        current_hash = _screenshot_hash()
        if current_hash and current_hash != initial_hash:
            time.sleep(0.3)
            return f"Screen changed after {time.time() - start:.1f}s"
    return f"Timeout after {timeout}s"

def wait_for_screen_stable(timeout: float = 10.0, stable_duration: float = 1.0) -> str:
    start = time.time()
    last_hash = _screenshot_hash()
    stable_since = time.time()
    while time.time() - start < timeout:
        time.sleep(0.3)
        current_hash = _screenshot_hash()
        if current_hash != last_hash:
            stable_since = time.time()
            last_hash = current_hash
        elif time.time() - stable_since >= stable_duration:
            return f"Screen stable for {stable_duration}s (wait: {time.time() - start:.1f}s)"
    return f"Timeout after {timeout}s"


# ── True DOM Extraction (CDP) ────────────────────────────────────────────────

def read_true_dom(max_chars: int = 40000) -> str:
    """Extracts a clean, interactable DOM tree via raw CDP."""
    global INTERACTABLE_MAP
    try:
        # Connect to the local debugging port
        req = urllib.request.Request("http://127.0.0.1:9222/json")
        with urllib.request.urlopen(req) as resp:
            targets = json.loads(resp.read())
            
        # Find the active page websocket
        ws_url = next((t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page' and not t['url'].startswith('devtools://')), None)
        if not ws_url:
            return "Error: No active Chromium page found. Launch browser first."

        # The JS payload builds a tree and calculates OS-level coordinates
        js_payload = """
        function getInteractableDOM() {
            let interactables = {};
            let idCounter = 1;
            
            const uiOffsetY = (window.outerHeight - window.innerHeight) || 85; 
            const uiOffsetX = (window.outerWidth - window.innerWidth) / 2 || 0;

            function isVisible(elem) {
                const rect = elem.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0 && window.getComputedStyle(elem).visibility !== 'hidden';
            }

            function traverse(node, depth) {
                if (node.nodeType === Node.TEXT_NODE) {
                    let text = node.textContent.trim();
                    if (text) return "  ".repeat(depth) + text + "\\n";
                    return "";
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return "";
                if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'PATH', 'META', 'LINK'].includes(node.tagName)) return "";
                if (!isVisible(node)) return "";

                let isInteractable = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(node.tagName) || 
                                     node.onclick != null || node.getAttribute('role') === 'button';

                let prefix = "";
                if (isInteractable) {
                    const rect = node.getBoundingClientRect();
                    const absX = Math.round(rect.left + rect.width / 2 + window.screenX + uiOffsetX);
                    const absY = Math.round(rect.top + rect.height / 2 + window.screenY + uiOffsetY);
                    
                    let label = node.tagName.toLowerCase();
                    if (node.id) label += `#${node.id}`;
                    if (node.name) label += `[name="${node.name}"]`;
                    if (node.placeholder) label += `[placeholder="${node.placeholder}"]`;
                    let text = node.innerText ? node.innerText.trim().substring(0, 30) : "";
                    if (text) label += ` "${text}"`;

                    interactables[idCounter] = { x: absX, y: absY, label: label };
                    prefix = `[ID: ${idCounter}] `;
                    idCounter++;
                }

                let str = "  ".repeat(depth) + prefix + `<${node.tagName.toLowerCase()}`;
                if (node.id) str += ` id="${node.id}"`;
                if (node.name) str += ` name="${node.name}"`;
                if (node.placeholder) str += ` placeholder="${node.placeholder}"`;
                if (node.value && node.tagName !== 'INPUT') str += ` value="${node.value}"`;
                if (node.href) str += ` href="${node.href}"`;
                str += ">\\n";

                for (let child of node.childNodes) {
                    str += traverse(child, depth + 1);
                }
                return str;
            }
            
            let domTree = traverse(document.body, 0);
            return JSON.stringify({ tree: domTree, map: interactables });
        }
        getInteractableDOM();
        """

        ws = websocket.create_connection(ws_url, timeout=5)
        msg = json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": js_payload, "returnByValue": True}
        })
        ws.send(msg)
        
        result = ""
        while True:
            response = json.loads(ws.recv())
            if response.get("id") == 1:
                val = response.get("result", {}).get("result", {}).get("value", "{}")
                data = json.loads(val)
                result = data.get("tree", "")
                INTERACTABLE_MAP = data.get("map", {})
                break
        ws.close()
        
        if len(result) > max_chars:
            return result[:max_chars] + "\\n...[DOM TRUNCATED]..."
        return result

    except Exception as e:
        return f"CDP Connection Error: {e}"


# ── Mouse & Keyboard Control ─────────────────────────────────────────────────

def get_mouse_position() -> Tuple[int, int]:
    return pyautogui.position()

def mouse_move(x: int, y: int, human: bool = True) -> str:
    x = max(0, min(x, SCREEN_WIDTH - 1))
    y = max(0, min(y, SCREEN_HEIGHT - 1))
    if human:
        duration = random.uniform(0.15, 0.35)
        pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeOutQuad)
    else:
        pyautogui.moveTo(x, y)
    return f"Mouse moved to ({x}, {y})"

def mouse_click(x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> str:
    if x is not None and y is not None:
        mouse_move(x, y, human=True)
        time.sleep(random.uniform(0.04, 0.10))
    pyautogui.click(button=button)
    time.sleep(0.12)
    return f"Clicked ({x}, {y}) [{button}]" if x else f"Clicked current [{button}]"

def mouse_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    if x is not None and y is not None:
        mouse_move(x, y, human=True)
        time.sleep(random.uniform(0.04, 0.10))
    pyautogui.doubleClick()
    time.sleep(0.12)
    return f"Double-clicked ({x}, {y})" if x else "Double-clicked current"

def mouse_scroll(direction: str = "down", clicks: int = 3) -> str:
    amount = -clicks if direction.lower() == "down" else clicks
    pyautogui.scroll(amount * 3)
    time.sleep(0.12)
    return f"Scrolled {direction} ({clicks} clicks)"

def mouse_drag(from_x: int, from_y: int, to_x: int, to_y: int) -> str:
    mouse_move(from_x, from_y, human=True)
    time.sleep(0.1)
    duration = random.uniform(0.3, 0.6)
    pyautogui.dragTo(to_x, to_y, duration=duration, button="left")
    return f"Dragged ({from_x},{from_y}) -> ({to_x},{to_y})"

def keyboard_type(text: str, human: bool = True) -> str:
    try:
        delay = str(random.randint(20, 60) if human else 5)
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", delay, "--", text],
            timeout=30, capture_output=True,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
    except Exception:
        pyautogui.write(text, interval=random.uniform(0.02, 0.06) if human else 0.0)
    return f"Typed: '{text[:50]}...'"

def keyboard_press(key: str) -> str:
    keys = [k.strip() for k in key.lower().split("+")]
    key_map = {"escape": "esc", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right", "return": "enter", "delete": "del"}
    keys = [key_map.get(k, k) for k in keys]
    pyautogui.hotkey(*keys)
    time.sleep(random.uniform(0.04, 0.10))
    return f"Pressed: {key}"

def keyboard_shortcut(keys: str) -> str:
    return keyboard_press(keys)


# ── Clipboard ────────────────────────────────────────────────────────────────

def clipboard_copy() -> str:
    keyboard_press("ctrl+c")
    time.sleep(0.2)
    return "Copied to clipboard"

def clipboard_paste() -> str:
    keyboard_press("ctrl+v")
    time.sleep(0.2)
    return "Pasted from clipboard"

def clipboard_get() -> str:
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        return result.stdout
    except Exception as e:
        return f"Clipboard read error: {e}"

def clipboard_set(text: str) -> str:
    try:
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        process.communicate(input=text.encode(), timeout=5)
        return f"Clipboard set: '{text[:50]}...'"
    except Exception as e:
        return f"Clipboard write error: {e}"


# ── Chromium Management ──────────────────────────────────────────────

def is_chromium_running() -> bool:
    try:
        result = subprocess.run(["pgrep", "-f", "chromium"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def _clean_chromium_locks():
    for pattern in ["SingletonLock", "SingletonCookie", "SingletonSocket"]:
        lock = Path(CHROMIUM_PROFILE_DIR) / pattern
        if lock.exists():
            try: lock.unlink()
            except Exception: pass
        for sub in Path(CHROMIUM_PROFILE_DIR).glob(f"*/{pattern}"):
            try: sub.unlink()
            except Exception: pass

def launch_chromium(url: str = "", profile: str = "") -> str:
    if is_chromium_running():
        if url:
            subprocess.Popen(
                ["chromium", "--no-sandbox", url],
                env={**os.environ, "DISPLAY": DISPLAY},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            wait_for_screen_change(timeout=5)
            return f"Opened new tab: {url}"
        return "Chromium is already running"

    _clean_chromium_locks()

    args = [
        "chromium",
        "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
        "--no-first-run", "--no-default-browser-check",
        "--disable-default-apps", "--disable-popup-blocking",
        "--disable-translate", "--start-maximized",
        "--disable-blink-features=AutomationControlled",
        "--remote-debugging-port=9222", 
        "--remote-allow-origins=*",
        "--password-store=basic",
        f"--user-data-dir={CHROMIUM_PROFILE_DIR}",
    ]

    if profile:
        profile_dir = Path(CHROMIUM_PROFILE_DIR) / f"Profile_{profile}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        args.append(f"--profile-directory=Profile_{profile}")

    if url:
        args.append(url)

    log_file = open("/tmp/chromium.log", "a")
    subprocess.Popen(args, env={**os.environ, "DISPLAY": DISPLAY}, stdout=log_file, stderr=log_file)

    wait_for_screen_stable(timeout=10, stable_duration=2.0)
    return f"Chromium launched | profile: {profile} | url: {url}"

def close_chromium() -> str:
    try:
        subprocess.run(["pkill", "-f", "chromium"], timeout=5, capture_output=True)
        time.sleep(1)
        subprocess.run(["pkill", "-9", "-f", "chromium"], timeout=5, capture_output=True)
        _clean_chromium_locks()
        return "Chromium closed"
    except Exception as e:
        return f"Error closing Chromium: {e}"

def chromium_navigate(url: str) -> str:
    keyboard_press("ctrl+l")
    time.sleep(0.4)
    keyboard_type(url, human=False)
    time.sleep(0.15)
    keyboard_press("enter")
    wait_for_screen_stable(timeout=12, stable_duration=1.5)
    return f"Navigated to: {url}"

def chromium_new_tab(url: str = "") -> str:
    keyboard_press("ctrl+t")
    time.sleep(0.5)
    if url:
        keyboard_type(url, human=False)
        time.sleep(0.15)
        keyboard_press("enter")
        wait_for_screen_stable(timeout=12, stable_duration=1.5)
    return f"New tab: {url}"

def chromium_close_tab() -> str:
    keyboard_press("ctrl+w")
    time.sleep(0.5)
    return "Tab closed"

def chromium_switch_tab(direction: str = "next") -> str:
    if direction == "next":
        keyboard_press("ctrl+tab")
    else:
        keyboard_press("ctrl+shift+tab")
    time.sleep(0.5)
    return f"Switched to {direction} tab"

def chromium_refresh() -> str:
    keyboard_press("f5")
    wait_for_screen_stable(timeout=12, stable_duration=1.5)
    return "Page refreshed"

def chromium_back() -> str:
    keyboard_press("alt+left")
    wait_for_screen_change(timeout=5)
    return "Navigated back"

def chromium_forward() -> str:
    keyboard_press("alt+right")
    wait_for_screen_change(timeout=5)
    return "Navigated forward"

def list_chromium_profiles() -> str:
    chromium_dir = Path(CHROMIUM_PROFILE_DIR)
    if not chromium_dir.exists():
        return "No profiles found."
    lines = ["Chromium Profiles:"]
    for d in sorted(chromium_dir.iterdir()):
        if d.is_dir() and d.name.startswith("Profile_"):
            lines.append(f"  {d.name.replace('Profile_', '')}")
    if (chromium_dir / "Default").exists():
        lines.append("  default")
    return "\n".join(lines)

def wipe_chromium_profile(profile: str) -> str:
    if profile.upper() == "ALL":
        shutil.rmtree(CHROMIUM_PROFILE_DIR, ignore_errors=True)
        return "All Chromium profiles wiped"
    profile_dir = Path(CHROMIUM_PROFILE_DIR) / f"Profile_{profile}"
    if profile_dir.exists():
        shutil.rmtree(profile_dir, ignore_errors=True)
        return f"Profile '{profile}' wiped"
    return f"Profile '{profile}' not found"


# ── Window Management ────────────────────────────────────────────────────────

def _xdotool(*args) -> str:
    try:
        result = subprocess.run(
            ["xdotool"] + list(args), capture_output=True, text=True, timeout=10,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        return result.stdout.strip()
    except Exception as e:
        return ""

def get_active_window() -> str:
    return _xdotool("getactivewindow", "getwindowname")

def focus_window(title_pattern: str) -> str:
    window_id = _xdotool("search", "--name", title_pattern)
    if window_id:
        _xdotool("windowactivate", window_id.splitlines()[0])
        time.sleep(0.3)
        return f"Focused window: {title_pattern}"
    return f"Window not found: {title_pattern}"