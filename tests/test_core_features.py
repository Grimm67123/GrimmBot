import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the core modules from your GrimmBot architecture
from core import AgentConfig
from core import CustomToolRegistry
from tools import Tools
from agent import GrimmAgent

# ── TEST 1: Unified JSON Custom Tool Creation ────────────────────────────────

def test_custom_tool_creation():
    """
    Verifies that create_tool securely stores Python code directly 
    inside custom_tools.json and correctly loads it into executable memory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the registry in a temporary sandbox
        registry = CustomToolRegistry(temp_dir)
        
        tool_name = "calculate_tax"
        tool_desc = "Multiplies a number by 0.2"
        tool_params = {
            "type": "object", 
            "properties": {"amount": {"type": "integer"}}
        }
        # The raw python string that will be injected into the JSON
        tool_code = """
def calculate_tax(amount):
    return amount * 0.2
"""
        # Execute the creation pipeline
        result_msg = registry.create_tool(tool_name, tool_desc, tool_params, tool_code)
        
        # Ensure active memory successfully bound the function
        assert "created" in result_msg
        assert tool_name in registry.list_tools()
        
        # Verify JSON physical storage
        json_path = Path(temp_dir) / "custom_tools.json"
        assert json_path.exists(), "custom_tools.json ledger was not generated."
        
        ledger_data = json.loads(json_path.read_text())
        assert len(ledger_data) == 1
        assert ledger_data[0]["name"] == tool_name
        assert ledger_data[0]["code"] == tool_code
        
        # Execute the dynamic tool call to prove it evaluates correctly
        execution_result = registry.call(tool_name, {"amount": 100})
        assert float(execution_result) == 20.0

# ── TEST 2: Active Error Analysis (Rule Appending) ───────────────────────────

def test_error_analysis_adaptation_save():
    """
    Verifies that when the agent invokes its save_adaptation_rule tool, 
    the rule is cleanly appended into the JSON physical file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the configuration paths to use our temporary test directory
        config = AgentConfig()
        config.adaptation_file = str(Path(temp_dir) / "adaptation.json")
        tools_module = Tools(config)
        
        test_rule = "Never use ping for external networking. Use curl -I instead."
        
        # The agent invokes the tool directly after encountering an error
        result_msg = tools_module.save_adaptation_rule(test_rule)
        
        assert "SUCCESS" in result_msg
        
        # Read the file directly from disk to confirm the write operation
        adap_path = Path(config.adaptation_file)
        assert adap_path.exists()
        
        saved_rules = json.loads(adap_path.read_text())
        assert len(saved_rules) == 1
        assert saved_rules[0] == test_rule

