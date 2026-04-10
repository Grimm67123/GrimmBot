"""
Tests for agent.py — GrimmAgent initialization, adaptation logic,
tool routing, approval system, emergency stop, and run_task flow.
All LLM calls are mocked — zero API cost.
"""

import os
import sys
import json
import pytest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from core import AgentConfig, init_safe_paths
from agent import GrimmAgent, TaskResult, StepLogger


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_llm_response(tool_calls=None, content=None):
    """Build a fake LLM completion response with optional tool calls."""
    msg = MagicMock()
    msg.content = content

    if tool_calls:
        tcs = []
        for tc_data in tool_calls:
            tc = MagicMock()
            tc.id = tc_data.get("id", "call_test123")
            tc.function.name = tc_data["name"]
            tc.function.arguments = json.dumps(tc_data.get("args", {}))
            tcs.append(tc)
        msg.tool_calls = tcs
    else:
        msg.tool_calls = None

    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in (msg.tool_calls or [])
        ]
    }

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
    return resp


@pytest.fixture
def test_config(tmp_path):
    """Provides a sterile configuration for testing."""
    cfg = AgentConfig()
    cfg.model = "test-model"
    cfg.vision_model = "test-vision-model"
    cfg.use_vision = True
    cfg.allowed_domains = ["github.com", "example.com"]
    cfg.allowed_commands = ["ls", "echo", "python", "curl"]
    cfg.task_timeout = 300
    cfg.max_iterations = 5
    cfg.wormhole_dir = str(tmp_path / "wormhole")
    cfg.workspace_dir = str(tmp_path / "workspace")
    cfg.data_dir = str(tmp_path / "data")
    cfg.profile_dir = str(tmp_path / "data" / "profiles")
    cfg.custom_tools_dir = str(tmp_path / "data" / "custom_tools")
    cfg.adaptation_file = str(tmp_path / "data" / "adaptation.json")
    cfg.memory_file = str(tmp_path / "data" / "memory.json")
    cfg.feedback_file = str(tmp_path / "data" / "feedback.json")
    cfg.settings_file = str(tmp_path / "data" / "settings.json")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# StepLogger Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStepLogger:
    def test_log_step_no_crash(self):
        logger = StepLogger()
        logger.log_step(1, "shell", {"command": "ls"}, "file1\nfile2")
        logger.log_step(2, "click", {"x": 100, "y": 200}, "OK")

    def test_log_thinking(self):
        logger = StepLogger()
        with patch.object(logger, "_broadcast") as mock_bc:
            logger.log_thinking("  I should use a tool.  ")
            mock_bc.assert_called_once_with("💭 **Thought Process:**\n> I should use a tool.")

    def test_log_error(self):
        logger = StepLogger()
        with patch.object(logger, "_broadcast") as mock_bc:
            logger.log_error("Failed to connect")
            mock_bc.assert_called_once_with("⚠️ **Agent Error:**\n`Failed to connect`")

    def test_fmt_args_click(self):
        logger = StepLogger()
        assert logger._fmt_args("click", {"x": 10, "y": 20}) == "10, 20"

    def test_fmt_args_type_text(self):
        logger = StepLogger()
        assert "Hello" in logger._fmt_args("type_text", {"text": "Hello World"})

    def test_fmt_args_shell(self):
        logger = StepLogger()
        assert "ls -la" in logger._fmt_args("shell", {"command": "ls -la"})

    def test_fmt_result_empty(self):
        logger = StepLogger()
        assert logger._fmt_result("") == ""
        assert logger._fmt_result(None) == ""

    def test_fmt_result_screenshot(self):
        logger = StepLogger()
        assert logger._fmt_result("Screenshot captured. Grid: ...") == ""

    def test_log_api_call_debug(self):
        logger = StepLogger()
        logger.debug_mode = True
        logger.log_api_call("gpt-4", 5)


# ══════════════════════════════════════════════════════════════════════════════
# Agent Init Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentInit:
    def test_basic_init(self, test_config):
        agent = GrimmAgent(test_config)
        assert agent.config.model == "test-model"
        assert agent.emergency_stop is False

    def test_tool_defs_include_vision_tools(self, test_config):
        test_config.use_vision = True
        agent = GrimmAgent(test_config)
        defs = agent._get_tool_defs()
        names = [d["function"]["name"] for d in defs]
        assert "screenshot" in names

    def test_tool_defs_exclude_vision_when_disabled(self, test_config):
        test_config.use_vision = False
        agent = GrimmAgent(test_config)
        defs = agent._get_tool_defs()
        names = [d["function"]["name"] for d in defs]
        assert "screenshot" not in names

    def test_creates_required_directories(self, test_config):
        GrimmAgent(test_config)
        assert os.path.exists(test_config.wormhole_dir)
        assert os.path.exists(test_config.workspace_dir)
        assert os.path.exists(test_config.custom_tools_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Adaptation Loading Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptationEngine:
    def test_adaptations_included_in_system_prompt(self, test_config):
        Path(test_config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(test_config.adaptation_file).write_text(json.dumps(["Never use cat for writes."]))

        agent = GrimmAgent(test_config)

        with patch("agent.completion") as mock_completion:
            mock_completion.return_value = make_llm_response(tool_calls=[{"name": "done", "args": {"result": "Done"}}])
            agent.run_task("Never use cat for writes test task")

            called_kwargs = mock_completion.call_args.kwargs
            system_prompt = next((m["content"] for m in called_kwargs["messages"] if m["role"] == "system"), "")

            assert "CRITICAL - LEARNED RULES" in system_prompt
            assert "Never use cat" in system_prompt

    def test_rag_filters_irrelevant_rules(self, test_config):
        """RAG should NOT inject rules that don't match the prompt keywords."""
        Path(test_config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(test_config.adaptation_file).write_text(json.dumps([
            "Never use ping to check network latency.",
            "Use curl instead of wget for downloads.",
        ]))

        agent = GrimmAgent(test_config)

        with patch("agent.completion") as mock_completion:
            mock_completion.return_value = make_llm_response(tool_calls=[{"name": "done", "args": {"result": "Done"}}])
            # This prompt is about Excel, not networking
            agent.run_task("Read an Excel spreadsheet file")

            called_kwargs = mock_completion.call_args.kwargs
            system_prompt = next((m["content"] for m in called_kwargs["messages"] if m["role"] == "system"), "")

            # Neither rule should be injected (they're about ping/curl/wget, not Excel)
            assert "ping" not in system_prompt
            assert "wget" not in system_prompt


# ══════════════════════════════════════════════════════════════════════════════
# Approval System Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestApprovalSystem:
    def test_shell_requires_approval(self, test_config):
        agent = GrimmAgent(test_config)
        assert agent._check_approval("shell", {"command": "ls"}) is True

        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("shell", {"command": "ls"}) is False

    def test_write_file_requires_approval(self, test_config):
        agent = GrimmAgent(test_config)
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("write_file", {"path": "t.txt"}) is False

    def test_click_no_approval_needed(self, test_config):
        agent = GrimmAgent(test_config)
        agent.commssafeguard = False
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("click", {"x": 1, "y": 2}) is True
        agent.approval_callback.assert_not_called()

    def test_go_to_url_blocked_domain_needs_approval(self, test_config):
        test_config.allowed_domains = ["google.com"]
        agent = GrimmAgent(test_config)
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("go_to_url", {"url": "https://google.com"}) is True
        assert agent._check_approval("go_to_url", {"url": "https://malicious.com"}) is False

    def test_commssafeguard_mode_type_text(self, test_config):
        agent = GrimmAgent(test_config)
        agent.commssafeguard = True
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("type_text", {"text": "hello"}) is False

    def test_commssafeguard_mode_enter_key(self, test_config):
        agent = GrimmAgent(test_config)
        agent.commssafeguard = True
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("press_key", {"key": "enter"}) is False

    def test_no_callback_auto_approves(self, test_config):
        agent = GrimmAgent(test_config)
        agent.approval_callback = None
        assert agent._check_approval("shell", {"command": "ls"}) is True

    def test_custom_tool_requires_approval(self, test_config):
        agent = GrimmAgent(test_config)
        agent.custom_tools._functions["my_tool"] = lambda: "ok"
        agent.custom_tools._requires_approval["my_tool"] = True
        agent.approval_callback = MagicMock(return_value=False)
        assert agent._check_approval("my_tool", {}) is False


# ══════════════════════════════════════════════════════════════════════════════
# run_task Full Loop Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRunTask:
    @patch("agent.completion")
    def test_simple_done_response(self, mock_completion, test_config):
        mock_completion.return_value = make_llm_response(
            tool_calls=[{"name": "done", "args": {"result": "Task completed!"}}]
        )
        agent = GrimmAgent(test_config)
        result = agent.run_task("Say hello")
        assert result.answer == "Task completed!"
        assert result.steps == 1

    @patch("agent.completion")
    def test_text_only_response(self, mock_completion, test_config):
        mock_completion.return_value = make_llm_response(content="Final answer.")
        agent = GrimmAgent(test_config)
        result = agent.run_task("Q")
        assert "Final answer" in result.answer

    @patch("agent.completion")
    def test_emergency_stop(self, mock_completion, test_config):
        agent = GrimmAgent(test_config)
        agent.emergency_stop = True
        result = agent.run_task("Do")
        assert "Emergency stop" in result.answer

    @patch("agent.completion")
    def test_max_iterations_reached(self, mock_completion, test_config):
        test_config.max_iterations = 2
        mock_completion.return_value = make_llm_response(tool_calls=[{"name": "get_current_time", "args": {}}])
        agent = GrimmAgent(test_config)
        result = agent.run_task("Loop")
        assert "Max iterations" in result.answer

    @patch("agent.completion")
    def test_tool_execution_passes_result(self, mock_completion, test_config):
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "get_current_time", "args": {}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "Done"}}]),
        ]
        agent = GrimmAgent(test_config)
        result = agent.run_task("Time")
        assert result.steps == 2

    @patch("agent.completion")
    def test_approval_denied_sends_denied_to_llm(self, mock_completion, test_config):
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "shell", "args": {"command": "ls"}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        agent.approval_callback = lambda t, a: False
        result = agent.run_task("Run ls")
        assert result.answer == "OK"

    @patch("agent.completion")
    def test_unknown_tool_handled(self, mock_completion, test_config):
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "ghost", "args": {}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        result = agent.run_task("Ghost")
        assert result.answer == "OK"

    @patch("agent.completion")
    def test_task_timeout(self, mock_completion, test_config):
        test_config.task_timeout = 0
        mock_completion.return_value = make_llm_response(tool_calls=[{"name": "wait", "args": {"seconds": 1}}])
        agent = GrimmAgent(test_config)
        result = agent.run_task("Task")
        assert "timed out" in result.answer.lower()

    @patch("agent.completion")
    def test_api_error_returned(self, mock_completion, test_config):
        mock_completion.side_effect = Exception("API down")
        agent = GrimmAgent(test_config)
        result = agent.run_task("Fail")
        assert "LLM error" in result.answer

    @patch("agent.completion")
    def test_rate_limit_retries(self, mock_completion, test_config):
        test_config.max_iterations = 3
        mock_completion.side_effect = [
            Exception("rate_limit exceeded"),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        with patch("agent.time.sleep"):
            result = agent.run_task("Retry")
        assert result.answer == "OK"

    @patch("agent.completion")
    def test_screenshot_in_vision_mode(self, mock_completion, test_config):
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "screenshot", "args": {}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        result = agent.run_task("Shot")
        assert result.answer == "OK"

    @patch("agent.completion")
    def test_screenshot_rejected_in_text_mode(self, mock_completion, test_config):
        test_config.use_vision = False
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "screenshot", "args": {}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        result = agent.run_task("Shot")
        assert result.answer == "OK"

    @patch("agent.completion")
    def test_tool_error_handled(self, mock_completion, test_config):
        mock_completion.side_effect = [
            make_llm_response(tool_calls=[{"name": "read_file", "args": {"path": "/bad"}}]),
            make_llm_response(tool_calls=[{"name": "done", "args": {"result": "OK"}}]),
        ]
        agent = GrimmAgent(test_config)
        result = agent.run_task("Bad file")
        assert result.answer == "OK"


class TestAgentCustomToolRouting:
    def test_create_custom_tool_via_agent(self, test_config):
        agent = GrimmAgent(test_config)
        with patch.object(agent.custom_tools, "create_tool", return_value="Created") as mock_create:
            with patch("agent.completion") as mock_completion:
                mock_completion.side_effect = [
                    make_llm_response(tool_calls=[{"name": "create_custom_tool", "args": {"name": "test_tool", "code": "pass", "description": "D"}}]),
                    make_llm_response(tool_calls=[{"name": "done", "args": {"result": "Finished"}}]),
                ]
                agent.run_task("Create tool")
                mock_create.assert_called_once()

    def test_list_custom_tools_via_agent(self, test_config):
        agent = GrimmAgent(test_config)
        agent.custom_tools._functions["my_tool"] = lambda: "hi"
        with patch("agent.completion") as mock_completion:
            mock_completion.side_effect = [
                make_llm_response(tool_calls=[{"name": "list_custom_tools", "args": {}}]),
                make_llm_response(tool_calls=[{"name": "done", "args": {"result": "Listed"}}]),
            ]
            result = agent.run_task("List tools")
            assert result.answer == "Listed"

    def test_custom_tool_approval_trigger(self, test_config):
        agent = GrimmAgent(test_config)
        agent.custom_tools._functions["secret_tool"] = lambda: "shhh"
        agent.custom_tools._requires_approval["secret_tool"] = True
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        assert agent._check_approval("secret_tool", {}) is True
        assert "secret_tool" in called

    def test_custom_tool_no_approval_trigger(self, test_config):
        agent = GrimmAgent(test_config)
        agent.custom_tools._functions["fast_tool"] = lambda: "vroom"
        agent.custom_tools._requires_approval["fast_tool"] = False
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        assert agent._check_approval("fast_tool", {}) is True
        assert len(called) == 0


class TestLocalModelSupport:
    """Tests for local model API base configuration."""

    def test_api_base_passed_to_completion_kwargs(self, test_config):
        test_config.api_base = "http://localhost:11434"
        agent = GrimmAgent(test_config)
        kwargs = agent._build_completion_kwargs("ollama/gemma3:1b", [], [])
        assert kwargs["api_base"] == "http://localhost:11434"

    def test_api_key_passed_to_completion_kwargs(self, test_config):
        test_config.api_key = "sk-test-key"
        agent = GrimmAgent(test_config)
        kwargs = agent._build_completion_kwargs("openai/gpt-4", [], [])
        assert kwargs["api_key"] == "sk-test-key"

    def test_no_api_base_when_not_set(self, test_config):
        test_config.api_base = ""
        test_config.api_key = ""
        agent = GrimmAgent(test_config)
        kwargs = agent._build_completion_kwargs("gemini/gemini-1.5-flash", [], [])
        assert "api_base" not in kwargs
        assert "api_key" not in kwargs