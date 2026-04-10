"""
Tests specifically for host timezone and timestamp accuracy in memory.
"""

import pytest
from datetime import datetime
from memory import MemoryStore, MemoryConfig

def test_memory_timestamp_uses_host_local_time(tmp_path):
    """Verify that when a memory is saved, the timestamp matches the host's current local time."""
    config = MemoryConfig(enabled=True, memory_file=str(tmp_path / "memory.json"))
    store = MemoryStore("timezone_test", config=config)
    
    # Capture time just before and after
    before = datetime.now().strftime("%Y-%m-%d %H:%M")
    store.add("Timezone test task", "Result")
    after = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Check the entry in the store
    entry = store.entries[0]
    
    # The timestamp should be at or between 'before' and 'after'
    # (usually they will all be the same string unless it flips exactly on the minute)
    assert entry.timestamp in [before, after]
    print(f"Captured host timestamp: {entry.timestamp}")
