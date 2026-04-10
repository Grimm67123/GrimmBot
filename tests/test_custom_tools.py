"""
Tests for custom_tools.py — CustomToolRegistry CRUD and execution.
Highly robust validation covering all unified JSON structures.
"""

import os
import sys
import json
import pytest
from pathlib import Path

from core import CustomToolRegistry


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def registry(tmp_workspace):
    return CustomToolRegistry(str(tmp_workspace["tools"]))


# ══════════════════════════════════════════════════════════════════════════════
# Create Tool Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateTool:
    def test_create_simple_tool(self, registry):
        code = "def add_nums(a, b):\n    return str(int(a) + int(b))"
        result = registry.create_tool(
            "add_nums", "Adds two numbers",
            {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            code,
        )
        assert "created" in result.lower()
        assert "add_nums" in registry.list_tools()

    def test_create_tool_invalid_name_uppercase(self, registry):
        result = registry.create_tool("BadName", "desc", {}, "pass")
        assert "Invalid" in result

    def test_create_tool_invalid_name_starts_with_number(self, registry):
        result = registry.create_tool("1tool", "desc", {}, "pass")
        assert "Invalid" in result

    def test_create_tool_invalid_name_special_chars(self, registry):
        result = registry.create_tool("tool-name", "desc", {}, "pass")
        assert "Invalid" in result

    def test_create_tool_overwrites_existing(self, registry):
        registry.create_tool("ovr", "D1", {}, "def ovr(): return 1")
        registry.create_tool("ovr", "D2", {}, "def ovr(): return 2")
        assert registry.call("ovr", {}) == "2"

    def test_create_tool_bad_code(self, registry):
        result = registry.create_tool("bad", "D", {}, "def bad(:")
        assert "failed to load" in result.lower()

    def test_create_tool_requires_approval_flag(self, registry):
        registry.create_tool("appr", "D", {}, "def appr(): return 1", requires_approval=False)
        assert registry._requires_approval["appr"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Call Tool Execution Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCallTool:
    def test_call_with_args(self, registry):
        registry.create_tool("greet", "D", {}, "def greet(name):\n    return f'Hello {name}'")
        assert registry.call("greet", {"name": "Grimm"}) == "Hello Grimm"

    def test_call_nonexistent_tool(self, registry):
        assert "not found" in registry.call("ghost", {}).lower()

    def test_call_tool_that_raises(self, registry):
        registry.create_tool("failer", "D", {}, "def failer():\n    raise ValueError('boom')")
        assert "error" in registry.call("failer", {}).lower()

    def test_call_tool_wrong_args(self, registry):
        registry.create_tool("no_args", "D", {}, "def no_args():\n    return 'ok'")
        # Passing args to a function that takes none creates a TypeError.
        # This fixes the assertion issue against the string formatted wrapper.
        result = registry.call("no_args", {"x": 1})
        assert "error" in result.lower()
        assert "keyword argument" in result


# ══════════════════════════════════════════════════════════════════════════════
# List / Delete Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestListDeleteTools:
    def test_list_empty(self, registry):
        assert registry.list_tools() == []

    def test_list_after_create(self, registry):
        registry.create_tool("t1", "D", {}, "def t1(): pass")
        registry.create_tool("t2", "D", {}, "def t2(): pass")
        assert sorted(registry.list_tools()) == ["t1", "t2"]

    def test_delete_tool(self, registry):
        registry.create_tool("delme", "D", {}, "def delme(): pass")
        registry.delete_tool("delme")
        assert "delme" not in registry.list_tools()

    def test_delete_nonexistent(self, registry):
        registry.delete_tool("nothing")


# ══════════════════════════════════════════════════════════════════════════════
# JSON Definition Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestGetDefinitions:
    def test_definitions_empty(self, registry):
        assert registry.get_definitions() == []

    def test_definitions_after_create(self, registry):
        params = {"type": "object", "properties": {"x": {"type": "int"}}}
        registry.create_tool("def_test", "My Desc", params, "def def_test(x): pass")
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "def_test"
        assert defs[0]["function"]["description"] == "My Desc"
        assert defs[0]["function"]["parameters"] == params

    def test_definitions_returns_copy(self, registry):
        registry.create_tool("t", "D", {}, "def t(): pass")
        d1 = registry.get_definitions()
        d1.clear()
        assert len(registry.get_definitions()) == 1


# ══════════════════════════════════════════════════════════════════════════════
# JSON Disk Persistence Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    def test_manifest_created(self, registry, tmp_workspace):
        registry.create_tool("persist", "P", {}, "def persist():\n    return 'ok'")
        ledger = Path(tmp_workspace["tools"]) / "custom_tools.json"
        assert ledger.exists()
        data = json.loads(ledger.read_text())
        assert any(t["name"] == "persist" for t in data)

    def test_code_file_created(self, registry, tmp_workspace):
        registry.create_tool("mycode", "C", {}, "def mycode():\n    return 'hi'")
        ledger = Path(tmp_workspace["tools"]) / "custom_tools.json"
        assert ledger.exists()
        data = json.loads(ledger.read_text())
        tool = next((t for t in data if t["name"] == "mycode"), None)
        assert tool is not None
        assert "def mycode" in tool["code"]

    def test_reload_from_disk(self, tmp_workspace):
        reg1 = CustomToolRegistry(str(tmp_workspace["tools"]))
        reg1.create_tool("reloaded", "R", {}, "def reloaded():\n    return 'from_disk'")
        del reg1

        reg2 = CustomToolRegistry(str(tmp_workspace["tools"]))
        assert "reloaded" in reg2.list_tools()
        assert reg2.call("reloaded", {}) == "from_disk"

    def test_reload_from_disk_with_approval_flag(self, tmp_workspace):
        reg1 = CustomToolRegistry(str(tmp_workspace["tools"]))
        reg1.create_tool("tool_false", "desc", {}, "def tool_false():\n    return 'no'", requires_approval=False)
        del reg1

        reg2 = CustomToolRegistry(str(tmp_workspace["tools"]))
        assert "tool_false" in reg2.list_tools()
        assert reg2._requires_approval.get("tool_false") is False