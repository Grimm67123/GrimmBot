"""
Tests for config.py — AgentConfig, domain/command/path safety.
"""

import os
import sys
import pytest
from pathlib import Path

from core import AgentConfig, is_domain_allowed, is_command_allowed, is_path_safe, init_safe_paths


# ══════════════════════════════════════════════════════════════════════════════
# AgentConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentConfig:
    def test_from_env_loads_model(self):
        os.environ["LLM_MODEL"] = "gemini/gemini-2.5-flash"
        config = AgentConfig.from_env()
        assert config.model == "gemini/gemini-2.5-flash"

    def test_from_env_loads_domains(self):
        os.environ["ALLOWED_DOMAINS"] = "github.com, ycombinator.com , reddit.com"
        config = AgentConfig.from_env()
        assert "github.com" in config.allowed_domains
        assert "ycombinator.com" in config.allowed_domains
        assert "reddit.com" in config.allowed_domains

    def test_from_env_wildcard_domains(self):
        os.environ["ALLOWED_DOMAINS"] = "*"
        config = AgentConfig.from_env()
        assert config.allow_all_domains is True

    def test_from_env_loads_commands(self):
        os.environ["ALLOWED_COMMANDS"] = "ls,grep,python"
        config = AgentConfig.from_env()
        assert "ls" in config.allowed_commands
        assert "python" in config.allowed_commands

    def test_from_env_wildcard_commands(self):
        os.environ["ALLOWED_COMMANDS"] = "*"
        config = AgentConfig.from_env()
        assert config.allow_all_commands is True

    def test_from_env_defaults(self):
        os.environ.pop("LLM_MODEL", None)
        config = AgentConfig.from_env()
        assert config.model == "gemini/gemini-1.5-flash"  # default fallback
        assert config.max_iterations == 50 or config.max_iterations > 0

    def test_vision_disabled(self):
        os.environ["USE_VISION"] = "false"
        config = AgentConfig.from_env()
        assert config.use_vision is False
        os.environ["USE_VISION"] = "true"  # restore

    def test_wormhole_blocked_extensions(self):
        os.environ["WORMHOLE_BLOCKED_EXTENSIONS"] = ".exe,.bat,.sh"
        config = AgentConfig.from_env()
        assert ".exe" in config.wormhole_blocked_extensions
        assert ".bat" in config.wormhole_blocked_extensions
        assert ".sh" in config.wormhole_blocked_extensions


# ══════════════════════════════════════════════════════════════════════════════
# Domain Safety
# ══════════════════════════════════════════════════════════════════════════════


class TestDomainSafety:
    @pytest.fixture(autouse=True)
    def setup_config(self):
        self.config = AgentConfig(
            allowed_domains=["github.com", "example.com"],
            allow_all_domains=False,
        )

    def test_allowed_domain(self):
        assert is_domain_allowed("https://github.com/torvalds/linux", self.config) is True

    def test_subdomain_allowed(self):
        assert is_domain_allowed("https://api.github.com/v3/repos", self.config) is True

    def test_blocked_domain(self):
        assert is_domain_allowed("https://malicious.com", self.config) is False

    def test_empty_url(self):
        assert is_domain_allowed("", self.config) is False

    def test_invalid_url(self):
        assert is_domain_allowed("not-a-url", self.config) is False

    def test_similar_but_different_domain(self):
        """e.g. 'notgithub.com' should NOT match 'github.com'."""
        assert is_domain_allowed("https://notgithub.com", self.config) is False

    def test_allow_all_domains(self):
        wildcard = AgentConfig(allowed_domains=["*"], allow_all_domains=True)
        assert is_domain_allowed("https://anything.com", wildcard) is True

    def test_http_and_https(self):
        assert is_domain_allowed("http://example.com/page", self.config) is True
        assert is_domain_allowed("https://example.com/page", self.config) is True


# ══════════════════════════════════════════════════════════════════════════════
# Command Safety
# ══════════════════════════════════════════════════════════════════════════════


class TestCommandSafety:
    @pytest.fixture(autouse=True)
    def setup_config(self):
        self.config = AgentConfig(
            allowed_commands=["ls", "grep", "python", "cat", "echo"],
            allow_all_commands=False,
        )

    def test_allowed_command(self):
        assert is_command_allowed("ls -la", self.config) is True

    def test_allowed_with_args(self):
        assert is_command_allowed("python script.py --verbose", self.config) is True

    def test_blocked_command(self):
        assert is_command_allowed("rm -rf /", self.config) is False

    def test_semicolon_injection(self):
        assert is_command_allowed("ls ; rm -rf /", self.config) is False

    def test_pipe_injection(self):
        assert is_command_allowed("ls | cat /etc/passwd", self.config) is False

    def test_and_injection(self):
        assert is_command_allowed("ls && rm -rf /", self.config) is False

    def test_or_injection(self):
        assert is_command_allowed("ls || malicious", self.config) is False

    def test_backtick_injection(self):
        assert is_command_allowed("echo `whoami`", self.config) is False

    def test_dollar_paren_injection(self):
        assert is_command_allowed("echo $(whoami)", self.config) is False

    def test_dollar_brace_injection(self):
        assert is_command_allowed("echo ${HOME}", self.config) is False

    def test_newline_injection(self):
        assert is_command_allowed("echo hello\nrm -rf /", self.config) is False

    def test_allow_all_commands(self):
        wildcard = AgentConfig(allowed_commands=["*"], allow_all_commands=True)
        assert is_command_allowed("anything --goes", wildcard) is True

    def test_empty_command(self):
        # empty string → splits to no parts → base_cmd is "" → not in allowlist
        assert is_command_allowed("", self.config) is False


# ══════════════════════════════════════════════════════════════════════════════
# Path Safety
# ══════════════════════════════════════════════════════════════════════════════


class TestPathSafety:
    def test_safe_paths(self, test_config):
        init_safe_paths(test_config)
        wormhole = test_config.wormhole_dir
        assert is_path_safe(os.path.join(wormhole, "test.txt")) is True

    def test_workspace_safe(self, test_config):
        init_safe_paths(test_config)
        workspace = test_config.workspace_dir
        assert is_path_safe(os.path.join(workspace, "project", "main.py")) is True

    def test_unsafe_path(self, test_config):
        init_safe_paths(test_config)
        if sys.platform != "win32":
            assert is_path_safe("/etc/passwd") is False
            assert is_path_safe("/root/.ssh/id_rsa") is False
        else:
            assert is_path_safe("C:\\Windows\\System32\\config\\SAM") is False

    def test_traversal_attack(self, test_config):
        init_safe_paths(test_config)
        wormhole = test_config.wormhole_dir
        # Attempt to escape wormhole via ../
        traversal = os.path.join(wormhole, "..", "..", "etc", "passwd")
        assert is_path_safe(traversal) is False
