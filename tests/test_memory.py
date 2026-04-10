"""
Tests for memory.py — MemoryStore RAG search, FeedbackStore, adaptation retrieval.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from memory import (
    MemoryConfig, MemoryEntry, MemoryStore, FeedbackStore,
    extract_keywords, keyword_overlap_score, retrieve_relevant_rules,
    get_memory, reset_memory, get_feedback_store,
)


# ══════════════════════════════════════════════════════════════════════════════
# Keyword Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestKeywordEngine:
    def test_extract_keywords_basic(self):
        kw = extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "quick" in kw
        assert "brown" in kw
        assert "fox" in kw
        # Stop words should be removed
        assert "the" not in kw
        assert "over" not in kw

    def test_extract_keywords_empty(self):
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_extract_keywords_with_numbers(self):
        kw = extract_keywords("Use port 8080 for the API on localhost")
        assert "8080" in kw
        assert "api" in kw
        assert "localhost" in kw

    def test_keyword_overlap_score_full_match(self):
        score = keyword_overlap_score(["python", "script"], ["python", "script", "test"])
        assert score > 0

    def test_keyword_overlap_score_no_match(self):
        score = keyword_overlap_score(["python", "script"], ["java", "compile"])
        assert score == 0.0

    def test_keyword_overlap_score_empty(self):
        assert keyword_overlap_score([], ["test"]) == 0.0
        assert keyword_overlap_score(["test"], []) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Adaptation RAG Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptationRAG:
    def test_retrieve_relevant_rules_match(self):
        rules = [
            "Do not use ping. Use curl instead.",
            "Always use python3 not python.",
            "The browser profile directory is /home/grimmbot/.config/chromium",
        ]
        result = retrieve_relevant_rules(rules, "Check network latency using ping")
        assert any("ping" in r for r in result)

    def test_retrieve_relevant_rules_no_match(self):
        rules = [
            "Do not use ping. Use curl instead.",
            "Always use python3 not python.",
        ]
        result = retrieve_relevant_rules(rules, "What is the weather today?")
        assert len(result) == 0

    def test_retrieve_relevant_rules_empty(self):
        assert retrieve_relevant_rules([], "anything") == []

    def test_retrieve_relevant_rules_respects_max(self):
        rules = [f"Rule about topic_{i}" for i in range(20)]
        result = retrieve_relevant_rules(rules, "topic_0 topic_1 topic_2", max_rules=2)
        assert len(result) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# MemoryStore Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryStore:
    def test_add_and_search(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        store.add("Write python script", "Script created successfully", ["code"])
        store.add("Check weather", "Sunny 72F", ["weather"])

        results = store.search("python script")
        assert len(results) >= 1
        assert any("python" in e.task_summary.lower() or "script" in e.task_summary.lower() for e in results)

    def test_search_no_results(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        results = store.search("nonexistent topic xyz123")
        assert results == []

    def test_persistence(self, tmp_path):
        mem_file = str(tmp_path / "mem.json")
        cfg = MemoryConfig(memory_file=mem_file)
        store1 = MemoryStore("default", cfg)
        store1.add("Test persistence", "Works!", ["test"])

        store2 = MemoryStore("default", cfg)
        assert len(store2.entries) == 1
        assert store2.entries[0].task_summary == "Test persistence"

    def test_reset(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        store.add("Task", "Result")
        assert len(store.entries) == 1
        store.reset()
        assert len(store.entries) == 0

    def test_get_context(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        store.add("Deploy python app", "Deployed to port 8080", ["deploy"])
        ctx = store.get_context("python deployment")
        assert "Deployed" in ctx or "deploy" in ctx.lower()

    def test_get_context_empty(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        assert store.get_context("anything") == ""

    def test_disabled_memory(self, tmp_path):
        cfg = MemoryConfig(enabled=False, memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        store.add("Task", "Result")
        assert len(store.entries) == 0
        assert store.get_context("anything") == ""

    def test_summarization(self, tmp_path):
        cfg = MemoryConfig(max_entries=5, keep_after_summary=2, memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        for i in range(5):
            store.add(f"Task {i}", f"Result {i}")
        assert store.needs_summarization
        result = store.request_summarization()
        assert "Summarized" in result
        assert len(store.entries) == 2

    def test_stats(self, tmp_path):
        cfg = MemoryConfig(memory_file=str(tmp_path / "mem.json"))
        store = MemoryStore("default", cfg)
        store.add("Task", "Result")
        stats = store.stats()
        assert stats["entries"] == 1
        assert stats["profile"] == "default"

    def test_legacy_entries_without_keywords(self, tmp_path):
        """Entries from old format (with 'embedding' field) should load gracefully."""
        mem_file = tmp_path / "mem.json"
        legacy_data = {
            "default": {
                "entries": [{
                    "id": "abc123",
                    "timestamp": "2026-01-01 00:00",
                    "task_summary": "Old task",
                    "result_summary": "Old result",
                    "tags": [],
                    "embedding": [0.1, 0.2, 0.3],
                }],
                "summaries": [],
            }
        }
        mem_file.write_text(json.dumps(legacy_data))
        cfg = MemoryConfig(memory_file=str(mem_file))
        store = MemoryStore("default", cfg)
        assert len(store.entries) == 1
        assert store.entries[0].task_summary == "Old task"


# ══════════════════════════════════════════════════════════════════════════════
# FeedbackStore Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackStore:
    def test_add_feedback(self, tmp_path):
        store = FeedbackStore(str(tmp_path / "fb.json"))
        store.add("Hello", "Hi there!", "positive", "msg-1")
        assert store.count() == 1
        assert store.get_all()[0]["rating"] == "positive"

    def test_persistence(self, tmp_path):
        fb_file = str(tmp_path / "fb.json")
        store1 = FeedbackStore(fb_file)
        store1.add("Q", "A", "negative")

        store2 = FeedbackStore(fb_file)
        assert store2.count() == 1

    def test_export_jsonl(self, tmp_path):
        store = FeedbackStore(str(tmp_path / "fb.json"))
        store.add("Prompt 1", "Response 1", "positive")
        store.add("Prompt 2", "Response 2", "negative")

        out_path = str(tmp_path / "rlhf.jsonl")
        result = store.export_jsonl(out_path)
        assert "Exported 2" in result

        lines = Path(out_path).read_text().strip().splitlines()
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["prompt"] == "Prompt 1"
        assert "chosen" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["prompt"] == "Prompt 2"
        assert "rejected" in entry2

    def test_empty_export(self, tmp_path):
        store = FeedbackStore(str(tmp_path / "fb.json"))
        result = store.export_jsonl(str(tmp_path / "empty.jsonl"))
        assert "Exported 0" in result
