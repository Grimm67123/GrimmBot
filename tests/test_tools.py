"""
Tests for tools.py — Every built-in tool method.
All screen/subprocess calls are mocked via conftest.py.
"""

import os
import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core import AgentConfig, init_safe_paths
from tools import Tools


# ══════════════════════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tools(test_config):
    init_safe_paths(test_config)
    return Tools(test_config)


# ══════════════════════════════════════════════════════════════════════════════
# Time & Wait Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestTimingTools:
    def test_get_current_time(self, tools):
        result = tools.get_current_time()
        assert "Current time:" in result

    def test_wait(self, tools):
        start = time.time()
        result = tools.wait(seconds=0.1)
        assert "Waited" in result

    def test_wait_capped_at_30(self, tools):
        """wait() should cap at 30 seconds (we just check it returns)."""
        # We mock time.sleep to avoid actually sleeping
        with patch("tools.time.sleep") as mock_sleep:
            tools.wait(seconds=999)
            mock_sleep.assert_called_once_with(30)

    def test_wait_for_change(self, tools):
        result = tools.wait_for_change(timeout=1.0)
        assert isinstance(result, str)

    def test_wait_for_stable(self, tools):
        result = tools.wait_for_stable(timeout=1.0)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Mouse & Keyboard Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestInputTools:
    def test_click(self, tools):
        result = tools.click(100, 200)
        assert isinstance(result, str)

    def test_click_right_button(self, tools):
        result = tools.click(100, 200, button="right")
        assert isinstance(result, str)

    def test_double_click(self, tools):
        result = tools.double_click(300, 400)
        assert isinstance(result, str)

    def test_move_mouse(self, tools):
        result = tools.move_mouse(500, 600)
        assert isinstance(result, str)

    def test_drag(self, tools):
        result = tools.drag(0, 0, 100, 100)
        assert isinstance(result, str)

    def test_scroll_down(self, tools):
        result = tools.scroll(direction="down", amount=5)
        assert isinstance(result, str)

    def test_scroll_up(self, tools):
        result = tools.scroll(direction="up", amount=3)
        assert isinstance(result, str)

    def test_type_text(self, tools):
        result = tools.type_text("Hello GrimmBot")
        assert isinstance(result, str)

    def test_press_key(self, tools):
        result = tools.press_key("enter")
        assert isinstance(result, str)

    def test_hotkey(self, tools):
        result = tools.hotkey("ctrl+s")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Clipboard Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestClipboardTools:
    def test_copy(self, tools):
        result = tools.copy()
        assert isinstance(result, str)

    def test_paste(self, tools):
        result = tools.paste()
        assert isinstance(result, str)

    def test_get_clipboard(self, tools):
        result = tools.get_clipboard()
        assert isinstance(result, str)

    def test_set_clipboard(self, tools):
        result = tools.set_clipboard("test data")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Browser Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestBrowserTools:
    def test_open_browser(self, tools):
        result = tools.open_browser("https://github.com")
        assert isinstance(result, str)

    def test_close_browser(self, tools):
        result = tools.close_browser()
        assert isinstance(result, str)

    def test_go_to_url_allowed(self, tools):
        result = tools.go_to_url("https://github.com/torvalds/linux")
        assert "not in allowlist" not in result

    def test_go_to_url_blocked(self, tools):
        result = tools.go_to_url("https://malicious.com")
        assert "not in allowlist" in result.lower() or "Domain" in result

    def test_new_tab_allowed(self, tools):
        result = tools.new_tab("https://example.com")
        assert "not in allowlist" not in result

    def test_new_tab_blocked(self, tools):
        result = tools.new_tab("https://evil.com")
        assert "not in allowlist" in result.lower() or "Domain" in result

    def test_new_tab_empty_url(self, tools):
        result = tools.new_tab("")
        assert isinstance(result, str)

    def test_close_tab(self, tools):
        result = tools.close_tab()
        assert isinstance(result, str)

    def test_switch_tab_next(self, tools):
        result = tools.switch_tab("next")
        assert isinstance(result, str)

    def test_refresh_page(self, tools):
        result = tools.refresh_page()
        assert isinstance(result, str)

    def test_go_back(self, tools):
        result = tools.go_back()
        assert isinstance(result, str)

    def test_go_forward(self, tools):
        result = tools.go_forward()
        assert isinstance(result, str)

    def test_read_dom(self, tools):
        result = tools.read_dom()
        assert isinstance(result, str)

    def test_click_element_not_found(self, tools):
        result = tools.click_element(99999)
        assert "not found" in result.lower() or "Error" in result


# ══════════════════════════════════════════════════════════════════════════════
# Window Management Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestWindowTools:
    def test_get_active_window_title(self, tools):
        result = tools.get_active_window_title()
        assert isinstance(result, str)

    def test_focus_window_by_title(self, tools):
        result = tools.focus_window_by_title("Terminal")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# File System Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestFileTools:
    def test_read_file(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "test.txt"
        f.write_text("Hello world\nLine 2\nLine 3")
        result = tools.read_file(str(f))
        assert "Hello world" in result
        assert "test.txt" in result

    def test_read_file_not_found(self, tools, test_config):
        result = tools.read_file(os.path.join(test_config.wormhole_dir, "nonexistent.txt"))
        assert "Not found" in result

    def test_read_file_unsafe_path(self, tools):
        result = tools.read_file("/etc/shadow")
        assert "outside" in result.lower() or "Path" in result

    def test_read_file_too_large(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "big.bin"
        f.write_bytes(b"x" * 1_100_000)
        result = tools.read_file(str(f))
        assert "too large" in result.lower()

    def test_read_file_lines(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "numbered.txt"
        f.write_text("\n".join(f"Line {i}" for i in range(1, 21)))
        result = tools.read_file_lines(str(f), start_line=5, end_line=10)
        assert "Line 5" in result
        assert "Line 10" in result

    def test_write_file(self, tools, test_config):
        target = os.path.join(test_config.wormhole_dir, "output.txt")
        result = tools.write_file(target, "Written by GrimmBot")
        assert "Written" in result
        assert Path(target).read_text() == "Written by GrimmBot"

    def test_write_file_creates_parents(self, tools, test_config):
        target = os.path.join(test_config.wormhole_dir, "sub", "deep", "file.txt")
        result = tools.write_file(target, "deep write")
        assert "Written" in result
        assert Path(target).exists()

    def test_write_file_unsafe_path(self, tools):
        result = tools.write_file("/tmp/evil.txt", "bad data")
        assert "outside" in result.lower() or "Path" in result

    def test_write_file_blocked_extension(self, tools, test_config):
        target = os.path.join(test_config.wormhole_dir, "malware.exe")
        result = tools.write_file(target, "MZ...")
        assert "not allowed" in result.lower()

    def test_patch_file(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "patch_me.txt"
        f.write_text("Hello World\nfoo bar\nbaz")
        result = tools.patch_file(str(f), "foo bar", "FOO BAR")
        assert "Patched" in result
        assert "FOO BAR" in f.read_text()

    def test_patch_file_not_found(self, tools, test_config):
        result = tools.patch_file(os.path.join(test_config.wormhole_dir, "nope.txt"), "a", "b")
        assert "Not found" in result

    def test_patch_file_pattern_not_found(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "patch_nf.txt"
        f.write_text("Hello World")
        result = tools.patch_file(str(f), "NONEXISTENT", "replacement")
        assert "not found" in result.lower()

    def test_patch_file_all_occurrences(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "multi.txt"
        f.write_text("aaa bbb aaa ccc aaa")
        result = tools.patch_file(str(f), "aaa", "XXX", occurrence=0)
        assert "3" in result  # 3 occurrences
        assert f.read_text() == "XXX bbb XXX ccc XXX"

    def test_insert_at_line(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "insert.txt"
        f.write_text("line1\nline2\nline3\n")
        result = tools.insert_at_line(str(f), 2, "INSERTED")
        assert "Inserted" in result
        content = f.read_text()
        lines = content.splitlines()
        assert lines[1] == "INSERTED"

    def test_delete_lines(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "del.txt"
        f.write_text("A\nB\nC\nD\nE\n")
        result = tools.delete_lines(str(f), 2, 4)
        assert "Deleted" in result
        assert "B" not in f.read_text()
        assert "C" not in f.read_text()
        assert "D" not in f.read_text()

    def test_list_directory(self, tools, test_config):
        d = Path(test_config.wormhole_dir)
        (d / "file1.txt").write_text("a")
        (d / "file2.txt").write_text("b")
        (d / "subdir").mkdir(exist_ok=True)
        result = tools.list_directory(str(d))
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "[dir]" in result

    def test_delete_file(self, tools, test_config):
        f = Path(test_config.wormhole_dir) / "delete_me.txt"
        f.write_text("gone")
        result = tools.delete_file(str(f))
        assert "Deleted" in result
        assert not f.exists()

    def test_find_in_files(self, tools, test_config):
        d = Path(test_config.wormhole_dir)
        (d / "search1.txt").write_text("The quick brown fox\njumps over the lazy dog")
        (d / "search2.txt").write_text("No match here")
        result = tools.find_in_files(str(d), "quick")
        assert "quick" in result
        assert "search1.txt" in result


# ══════════════════════════════════════════════════════════════════════════════
# Shell Tool
# ══════════════════════════════════════════════════════════════════════════════


class TestShellTool:
    def test_shell_allowed_command(self, tools):
        with patch("tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="output\n", stderr=""
            )
            result = tools.shell("ls -la")
            assert "EXIT: 0" in result
            assert "output" in result

    def test_shell_blocked_command(self, tools):
        result = tools.shell("rm -rf /")
        assert "not in allowlist" in result.lower() or "blocked" in result.lower()

    def test_shell_injection_blocked(self, tools):
        result = tools.shell("ls ; rm -rf /")
        assert "not in allowlist" in result.lower() or "blocked" in result.lower()

    def test_shell_timeout(self, tools):
        with patch("tools.subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("ls", 120)
            result = tools.shell("ls")
            assert "timed out" in result.lower()

    def test_shell_stderr(self, tools):
        with patch("tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error: not found\n"
            )
            result = tools.shell("grep nonexistent")
            assert "STDERR" in result


# ══════════════════════════════════════════════════════════════════════════════
# Memory Tools (remember / recall)
# ══════════════════════════════════════════════════════════════════════════════


class TestMemoryTools:
    def test_remember(self, tools):
        with patch("tools.get_memory") as mock_mem:
            mock_mem.return_value = MagicMock()
            result = tools.remember("Important note", tags=["test"])
            assert "Remembered" in result

    def test_recall_no_results(self, tools):
        with patch("tools.get_memory") as mock_mem:
            mock_mem.return_value.search.return_value = []
            result = tools.recall("something")
            assert "No relevant" in result


# ══════════════════════════════════════════════════════════════════════════════
# Plan Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestPlanTools:
    def test_create_plan(self, tools, test_config):
        result = tools.create_plan("Test goal", ["Step 1", "Step 2", "Step 3"])
        assert "Plan created" in result
        assert "3 steps" in result
        plan_file = Path(test_config.workspace_dir) / "current_plan.md"
        assert plan_file.exists()
        content = plan_file.read_text()
        assert "Test goal" in content
        assert "Step 1" in content

    def test_update_plan_step(self, tools, test_config):
        tools.create_plan("Goal", ["Do thing 1", "Do thing 2"])
        result = tools.update_plan_step(1, status="done", notes="completed")
        assert "marked as done" in result

    def test_update_plan_step_not_found(self, tools, test_config):
        tools.create_plan("Goal", ["Step 1"])
        result = tools.update_plan_step(99)
        assert "not found" in result.lower()

    def test_update_plan_no_plan(self, tools):
        result = tools.update_plan_step(1)
        assert "No active plan" in result


# ══════════════════════════════════════════════════════════════════════════════
# Browser Profile Tools
# ══════════════════════════════════════════════════════════════════════════════


class TestProfileTools:
    def test_list_profiles(self, tools):
        result = tools.list_profiles()
        assert isinstance(result, str)

    def test_wipe_profile(self, tools):
        result = tools.wipe_profile("TestProfile")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Done Tool
# ══════════════════════════════════════════════════════════════════════════════


class TestDoneTool:
    def test_done_returns_result(self, tools):
        assert tools.done("Task completed successfully") == "Task completed successfully"
