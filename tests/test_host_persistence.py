"""
Tests for host-based persistence (memory.json and settings.json).
"""

import os
import json
import pytest
import shutil
from pathlib import Path
from agent import GrimmAgent, AgentConfig
from memory import MemoryStore, MemoryConfig

@pytest.fixture
def temp_run_dir(tmp_path):
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(orig_cwd)

def test_config_defaults(temp_run_dir):
    config = AgentConfig()
    # Verify that defaults are no longer /app/ paths
    assert config.wormhole_dir == "wormhole"
    assert config.workspace_dir == "workspace"
    assert config.profile_dir == "data/profiles"

def test_memory_json_persistence(temp_run_dir):
    # Setup memory config to use a local file
    memory_config = MemoryConfig(memory_file="memory.json")
    
    # Profile 'default'
    store1 = MemoryStore("default", config=memory_config)
    store1.add("Initial task", "Success")
    
    # Verify file exists in root
    assert Path("memory.json").exists()
    
    # Verify content structure
    data = json.loads(Path("memory.json").read_text())
    assert "default" in data
    assert data["default"]["entries"][0]["task_summary"] == "Initial task"
    
    # Verify retrieval in a new instance
    store2 = MemoryStore("default", config=memory_config)
    assert len(store2.entries) == 1
    assert store2.entries[0].result_summary == "Success"

def test_agent_settings_json_persistence(temp_run_dir):
    # Create required dirs so agent init doesn't fail on missing paths
    Path("wormhole").mkdir()
    Path("workspace").mkdir()
    Path("data/profiles").mkdir(parents=True)
    Path("data/custom_tools").mkdir(parents=True)
    
    config = AgentConfig(
        wormhole_dir="wormhole",
        workspace_dir="workspace",
        profile_dir="data/profiles",
        custom_tools_dir="data/custom_tools"
    )
    
    agent = GrimmAgent(config=config)
    # Patch settings file for test
    agent._settings_file = Path("settings.json")
    
    # Modify settings
    agent.throttle_seconds = 10
    agent.commssafeguard = True
    agent.verbose = True
    agent.save_settings()
    
    # Verify file exists
    assert Path("settings.json").exists()
    
    # Create new agent instance
    agent2 = GrimmAgent(config=config)
    agent2._settings_file = Path("settings.json")
    agent2._load_settings()
    
    # Verify settings reloaded
    assert agent2.throttle_seconds == 10
    assert agent2.commssafeguard is True
    assert agent2.verbose is True

def test_settings_malformed_json(temp_run_dir):
    """Verify that agent defaults to safe values if settings.json is corrupted."""
    Path("settings.json").write_text("{ NOT JSON }")
    config = AgentConfig(wormhole_dir="w", workspace_dir="w", profile_dir="p", custom_tools_dir="t")
    agent = GrimmAgent(config=config)
    
    # Should not crash and should use defaults (commssafeguard is True by default)
    assert agent.commssafeguard is True
    assert agent.throttle_seconds == 0

def test_memory_multi_profile_isolation(temp_run_dir):
    """Verify that multiple profiles can coexist in memory.json without overwriting."""
    memory_config = MemoryConfig(memory_file="memory.json")
    
    # Save to profile 'user1'
    store1 = MemoryStore("user1", config=memory_config)
    store1.add("User 1 task", "Done")
    
    # Save to profile 'user2'
    store2 = MemoryStore("user2", config=memory_config)
    store2.add("User 2 task", "Done")
    
    # Re-load 'user1' from same file
    store1_reload = MemoryStore("user1", config=memory_config)
    assert len(store1_reload.entries) == 1
    assert store1_reload.entries[0].task_summary == "User 1 task"
    
    # Re-load 'user2' from same file
    store2_reload = MemoryStore("user2", config=memory_config)
    assert len(store2_reload.entries) == 1
    assert store2_reload.entries[0].task_summary == "User 2 task"

