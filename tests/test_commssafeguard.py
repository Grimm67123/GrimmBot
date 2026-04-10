"""
Tests for commssafeguard — Smart detection of communication buttons.
"""

import pytest
from unittest.mock import patch, MagicMock
from agent import GrimmAgent, AgentConfig

@pytest.fixture
def test_config():
    config = AgentConfig(
        wormhole_dir="wormhole",
        workspace_dir="workspace",
        profile_dir="data",
        custom_tools_dir="tools"
    )
    return config

def test_commssafeguard_blocks_submit_button(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    # Mock INTERACTABLE_MAP with a submit button
    mock_map = {"1": {"x": 100, "y": 100, "label": "button#submit \"Submit Message\""}}
    
    with patch("screen.INTERACTABLE_MAP", mock_map):
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        
        # This should trigger approval
        agent._check_approval("click_element", {"element_id": 1})
        assert len(called) == 1
        assert called[0] == "click_element"

def test_commssafeguard_blocks_send_button(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    mock_map = {"5": {"x": 200, "y": 200, "label": "a#send-btn \"Send\""}}
    
    with patch("screen.INTERACTABLE_MAP", mock_map):
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        
        agent._check_approval("click_element", {"element_id": 5})
        assert len(called) == 1

def test_commssafeguard_allows_regular_link(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    mock_map = {"10": {"x": 300, "y": 300, "label": "a#home \"Home Page\""}}
    
    with patch("screen.INTERACTABLE_MAP", mock_map):
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        
        # This should NOT trigger approval
        agent._check_approval("click_element", {"element_id": 10})
        assert len(called) == 0

def test_commssafeguard_blocks_raw_clicks(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    called = []
    agent.approval_callback = lambda tool, args: (called.append(tool) or True)
    
    # Raw clicks are always suspicious when safeguard is ON
    agent._check_approval("click", {"x": 500, "y": 500})
    assert len(called) == 1

def test_commssafeguard_case_insensitivity(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    mock_map = {"7": {"x": 0, "y": 0, "label": "button#btn \"SEND MESSAGE\""}}
    
    with patch("screen.INTERACTABLE_MAP", mock_map):
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        
        agent._check_approval("click_element", {"element_id": 7})
        assert len(called) == 1

def test_commssafeguard_substring_match(test_config):
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    mock_map = {"8": {"x": 0, "y": 0, "label": "div \"post comment now\""}}
    
    with patch("screen.INTERACTABLE_MAP", mock_map):
        called = []
        agent.approval_callback = lambda tool, args: (called.append(tool) or True)
        
        agent._check_approval("click_element", {"element_id": 8})
        assert len(called) == 1

def test_commssafeguard_missing_id_graceful(test_config):
    """Verify that agent doesn't crash if an element ID is missing from the map."""
    agent = GrimmAgent(test_config)
    agent.commssafeguard = True
    
    with patch("screen.INTERACTABLE_MAP", {}):
        # Should not crash, just not trigger approval unless it matches other rules
        res = agent._check_approval("click_element", {"element_id": 999})
        assert res is True

