"""
Shared fixtures for GrimmBot test suite.
All external I/O (LLM API, screen, subprocess) is mocked here.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import datetime


# ── Ensure project root is on sys.path ────────────────────────────────────────

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Environment Setup ─────────────────────────────────────────────────────────

# Set dummy env vars BEFORE importing any project modules
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("ALLOWED_DOMAINS", "github.com,example.com")
os.environ.setdefault("ALLOWED_COMMANDS", "ls,grep,python,cat,echo,find")
os.environ.setdefault("USE_VISION", "true")
os.environ.setdefault("SESSION_ENCRYPTION_KEY", "test-encryption-key-1234567890")
os.environ.setdefault("MEMORY_ENABLED", "true")
os.environ.setdefault("TIMEZONE", "UTC")


# ── Mock External Dependencies for Windows/Local Testing ──────────────────────

class MockLiteLLM(MagicMock): pass

class MockWebsocket(MagicMock): pass

class MockPyAutoGUI(MagicMock):
    FAILSAFE = False
    PAUSE = 0.03

class MockLinalg:
    @staticmethod
    def norm(vec):
        import math
        return math.sqrt(sum(x*x for x in vec))

class MockNumpyArray(list):
    def tolist(self): return self
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockNumpyArray([x / other for x in self])
        return NotImplemented

class MockNumpy:
    float64 = float
    linalg = MockLinalg()
    @staticmethod
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    @staticmethod
    def zeros(shape, dtype=float):
        if isinstance(shape, tuple): shape = shape[0]
        return MockNumpyArray([0.0] * shape)
    @staticmethod
    def array(a):
        return MockNumpyArray(a)

class MockPytz:
    UTC = datetime.timezone.utc
    @staticmethod
    def timezone(zone):
        return datetime.timezone.utc

class MockFernet:
    def __init__(self, key):
        self.key = key
    def encrypt(self, data):
        return b"enc:" + self.key + b":" + data
    def decrypt(self, data):
        prefix = b"enc:" + self.key + b":"
        if not data.startswith(prefix):
            from cryptography.fernet import InvalidToken
            raise InvalidToken()
        return data[len(prefix):]

class MockFernetModule:
    Fernet = MockFernet

class MockCryptography:
    fernet = MockFernetModule()

# Inject mocks so imports in agent.py, memory.py, etc. don't fail parsing or testing logic
sys.modules['litellm'] = MockLiteLLM()
sys.modules['numpy'] = MockNumpy()
sys.modules['pytz'] = MockPytz()
sys.modules['websocket'] = MockWebsocket()
sys.modules['pyautogui'] = MockPyAutoGUI()
sys.modules['Xlib'] = MagicMock()
sys.modules['Xlib.display'] = MagicMock()

# Since cryptography structure is used directly by memory.py we need standard Python classes
class MockInvalidToken(Exception): pass
sys.modules['cryptography'] = MockCryptography()
sys.modules['cryptography.fernet'] = MockFernetModule()
sys.modules['cryptography.fernet'].InvalidToken = MockInvalidToken


# ── Temp Directory Fixture ────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path):
    """Creates a temporary workspace with wormhole, workspace, data, and tools dirs."""
    dirs = {
        "wormhole": tmp_path / "wormhole",
        "workspace": tmp_path / "workspace",
        "data": tmp_path / "data",
        "tools": tmp_path / "data" / "custom_tools",
        "profiles": tmp_path / "data" / "profiles",
        "memory": tmp_path / "data",
        "scheduler": tmp_path / "data" / "scheduler",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def test_config(tmp_workspace):
    """AgentConfig pointing at temp directories."""
    from core import AgentConfig
    return AgentConfig(
        model="test-model",
        vision_model="test-vision-model",
        use_vision=True,
        allowed_domains=["github.com", "example.com"],
        allow_all_domains=False,
        allowed_commands=["ls", "grep", "python", "cat", "echo", "find"],
        allow_all_commands=False,
        task_timeout=60,
        max_iterations=10,
        max_shell_output=4000,
        wormhole_dir=str(tmp_workspace["wormhole"]),
        workspace_dir=str(tmp_workspace["workspace"]),
        data_dir=str(tmp_workspace["data"]),
        profile_dir=str(tmp_workspace["profiles"]),
        custom_tools_dir=str(tmp_workspace["tools"]),
        adaptation_file=str(tmp_workspace["data"] / "adaptation.json"),
        memory_file=str(tmp_workspace["data"] / "memory.json"),
        feedback_file=str(tmp_workspace["data"] / "feedback.json"),
        settings_file=str(tmp_workspace["data"] / "settings.json"),
        wormhole_max_file_size=1048576,
        wormhole_blocked_extensions=[".exe", ".bat"],
    )


# ── Screen Mock ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_screen():
    """Auto-mock all screen.py functions so tests never touch X11/pyautogui."""
    screen_mocks = {
        "screen.take_screenshot_raw": MagicMock(return_value=b"fake-png-bytes"),
        "screen.screenshot_to_base64": MagicMock(return_value="ZmFrZQ=="),
        "screen.save_screenshot": MagicMock(return_value="saved"),
        "screen.mouse_move": MagicMock(return_value="Moved to (100, 200)"),
        "screen.mouse_click": MagicMock(return_value="Clicked at (100, 200)"),
        "screen.mouse_double_click": MagicMock(return_value="Double-clicked at (100, 200)"),
        "screen.mouse_scroll": MagicMock(return_value="Scrolled down 3"),
        "screen.mouse_drag": MagicMock(return_value="Dragged (0,0) -> (100,100)"),
        "screen.keyboard_type": MagicMock(return_value="Typed text"),
        "screen.keyboard_press": MagicMock(return_value="Pressed enter"),
        "screen.keyboard_shortcut": MagicMock(return_value="Pressed ctrl+s"),
        "screen.clipboard_copy": MagicMock(return_value="Copied"),
        "screen.clipboard_paste": MagicMock(return_value="Pasted"),
        "screen.clipboard_get": MagicMock(return_value="clipboard content"),
        "screen.clipboard_set": MagicMock(return_value="Clipboard set"),
        "screen.launch_chromium": MagicMock(return_value="Browser launched"),
        "screen.close_chromium": MagicMock(return_value="Browser closed"),
        "screen.chromium_navigate": MagicMock(return_value="Navigated"),
        "screen.chromium_new_tab": MagicMock(return_value="New tab opened"),
        "screen.chromium_close_tab": MagicMock(return_value="Tab closed"),
        "screen.chromium_switch_tab": MagicMock(return_value="Switched tab"),
        "screen.chromium_refresh": MagicMock(return_value="Refreshed"),
        "screen.chromium_back": MagicMock(return_value="Went back"),
        "screen.chromium_forward": MagicMock(return_value="Went forward"),
        "screen.is_chromium_running": MagicMock(return_value=False),
        "screen.list_chromium_profiles": MagicMock(return_value="Profiles: Default"),
        "screen.wipe_chromium_profile": MagicMock(return_value="Wiped"),
        "screen.get_active_window": MagicMock(return_value="Terminal"),
        "screen.focus_window": MagicMock(return_value="Focused"),
        "screen.wait_for_screen_change": MagicMock(return_value="Screen changed"),
        "screen.wait_for_screen_stable": MagicMock(return_value="Screen stable"),
        "screen.read_true_dom": MagicMock(return_value="<html><body>Test page</body></html>"),

        # Patch them in tools to avoid 'from screen import func' binding bypass
        "tools.launch_chromium": MagicMock(return_value="Browser launched"),
        "tools.close_chromium": MagicMock(return_value="Browser closed"),
        "tools.chromium_navigate": MagicMock(return_value="Navigated"),
        "tools.read_true_dom": MagicMock(return_value="<html><body>Test page</body></html>"),
    }
    patches = [patch(target, mock) for target, mock in screen_mocks.items()]
    for p in patches:
        p.start()
    yield screen_mocks
    for p in patches:
        p.stop()
