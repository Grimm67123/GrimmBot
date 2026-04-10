"""
Tests for core.py — AgentConfig, Safety, CustomToolRegistry, Tool Definitions.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from core import (
    AgentConfig, init_safe_paths, is_path_safe,
    is_domain_allowed, is_command_allowed,
    CustomToolRegistry, TOOL_DEFINITIONS,
)


# ══════════════════════════════════════════════════════════════════════════════
# AgentConfig Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == ""
        assert cfg.use_vision is True
        assert cfg.max_iterations == 50
        assert cfg.data_dir == "data"
        assert cfg.api_base == ""
        assert cfg.api_key == ""

    def test_from_env_basic(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "gemini/gemini-1.5-flash",
            "ALLOWED_DOMAINS": "google.com,github.com",
            "ALLOWED_COMMANDS": "ls,echo",
            "PROVIDER_API_BASE": "",
        }, clear=False):
            cfg = AgentConfig.from_env()
            assert cfg.model == "gemini/gemini-1.5-flash"
            assert "google.com" in cfg.allowed_domains
            assert "ls" in cfg.allowed_commands

    def test_from_env_ollama_auto_detect(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "ollama/gemma3:1b",
            "PROVIDER_API_BASE": "",
        }, clear=False):
            cfg = AgentConfig.from_env()
            assert cfg.api_base == "http://host.docker.internal:11434"

    def test_from_env_explicit_api_base(self):
        with patch.dict(os.environ, {
            "LLM_MODEL": "openai/gpt-4",
            "PROVIDER_API_BASE": "http://localhost:1234/v1",
        }, clear=False):
            cfg = AgentConfig.from_env()
            assert cfg.api_base == "http://localhost:1234/v1"

    def test_wildcard_domains(self):
        with patch.dict(os.environ, {
            "ALLOWED_DOMAINS": "*",
        }, clear=False):
            cfg = AgentConfig.from_env()
            assert cfg.allow_all_domains is True


# ══════════════════════════════════════════════════════════════════════════════
# Safety Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSafety:
    def test_is_path_safe(self, tmp_path):
        safe = str(tmp_path / "workspace")
        Path(safe).mkdir(exist_ok=True)
        init_safe_paths(AgentConfig(workspace_dir=safe, wormhole_dir=safe, custom_tools_dir=safe))
        assert is_path_safe(str(tmp_path / "workspace" / "file.txt")) is True
        assert is_path_safe("/etc/passwd") is False

    def test_is_domain_allowed(self):
        cfg = AgentConfig(allowed_domains=["github.com", "google.com"])
        assert is_domain_allowed("https://github.com/repo", cfg) is True
        assert is_domain_allowed("https://evil.com/hack", cfg) is False
        assert is_domain_allowed("https://api.github.com/v3", cfg) is True

    def test_is_domain_allowed_wildcard(self):
        cfg = AgentConfig(allow_all_domains=True)
        assert is_domain_allowed("https://anything.com", cfg) is True

    def test_is_command_allowed(self):
        cfg = AgentConfig(allowed_commands=["ls", "echo", "python"])
        assert is_command_allowed("ls -la", cfg) is True
        assert is_command_allowed("rm -rf /", cfg) is False
        assert is_command_allowed("echo hello", cfg) is True

    def test_is_command_blocked_chars(self):
        cfg = AgentConfig(allowed_commands=["ls"])
        assert is_command_allowed("ls; rm -rf /", cfg) is False
        assert is_command_allowed("ls && echo pwned", cfg) is False
        assert is_command_allowed("ls | grep foo", cfg) is False

    def test_is_command_wildcard(self):
        cfg = AgentConfig(allow_all_commands=True)
        assert is_command_allowed("anything_goes", cfg) is True


# ══════════════════════════════════════════════════════════════════════════════
# CustomToolRegistry Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomToolRegistry:
    def test_create_and_call(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        result = registry.create_tool(
            name="add_numbers",
            description="Add two numbers",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            code='def add_numbers(a, b):\n    return a + b',
        )
        assert "created" in result.lower() or "saved" in result.lower()
        assert registry.call("add_numbers", {"a": 3, "b": 5}) == "8"

    def test_list_tools(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        assert registry.list_tools() == []
        registry.create_tool("tool_a", "desc", {}, 'def tool_a(): return "a"')
        assert "tool_a" in registry.list_tools()

    def test_delete_tool(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        registry.create_tool("temp_tool", "desc", {}, 'def temp_tool(): return "x"')
        assert "temp_tool" in registry.list_tools()
        registry.delete_tool("temp_tool")
        assert "temp_tool" not in registry.list_tools()

    def test_invalid_name_rejected(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        result = registry.create_tool("123invalid", "desc", {}, "pass")
        assert "Invalid" in result

    def test_approval_setting(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        registry.create_tool("safe_tool", "desc", {}, 'def safe_tool(): return 1', requires_approval=True)
        assert registry._requires_approval["safe_tool"] is True
        registry.set_approval_requirement("safe_tool", False)
        assert registry._requires_approval["safe_tool"] is False

    def test_persistence_across_loads(self, tmp_path):
        tools_dir = str(tmp_path / "tools")
        reg1 = CustomToolRegistry(tools_dir)
        reg1.create_tool("persistent", "desc", {}, 'def persistent(): return "alive"')
        # Load a new registry from the same dir
        reg2 = CustomToolRegistry(tools_dir)
        assert "persistent" in reg2.list_tools()
        assert reg2.call("persistent", {}) == "alive"

    def test_get_definitions_format(self, tmp_path):
        registry = CustomToolRegistry(str(tmp_path / "tools"))
        registry.create_tool("my_func", "A test", {}, 'def my_func(): return 42')
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "my_func"


# ══════════════════════════════════════════════════════════════════════════════
# Tool Definitions Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    def test_all_definitions_have_required_fields(self):
        for td in TOOL_DEFINITIONS:
            assert "type" in td
            assert td["type"] == "function"
            assert "function" in td
            assert "name" in td["function"]
            assert "description" in td["function"]
            assert "parameters" in td["function"]

    def test_essential_tools_present(self):
        names = {td["function"]["name"] for td in TOOL_DEFINITIONS}
        essential = ["screenshot", "click", "shell", "read_file", "write_file",
                     "done", "remember", "recall", "save_adaptation_rule"]
        for name in essential:
            assert name in names, f"Missing essential tool: {name}"

    def test_no_duplicate_tool_names(self):
        names = [td["function"]["name"] for td in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), f"Duplicate tool names found"
