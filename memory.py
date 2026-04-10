"""
GrimmBot — Memory & Feedback Module
Persistent RAG memory with keyword-based retrieval (no vector embeddings).
Feedback store for RLHF fine-tuning data generation.
"""

import os
import re
import json
import math
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter

logger = logging.getLogger("memory")


# ══════════════════════════════════════════════════════════════════════════════
# Memory Configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_entries: int = 100
    keep_after_summary: int = 20
    memory_file: str = "data/memory.json"

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true",
            max_entries=int(os.getenv("MEMORY_MAX_ENTRIES", "100")),
            keep_after_summary=int(os.getenv("MEMORY_KEEP_AFTER_SUMMARY", "20")),
            memory_file=os.getenv("MEMORY_FILE", "data/memory.json"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Memory Entry
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryEntry:
    id: str
    timestamp: str
    task_summary: str
    result_summary: str
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        # Strip legacy 'embedding' field if present
        cleaned = {k: v for k, v in d.items() if k in valid}
        return cls(**cleaned)

    def to_text(self) -> str:
        return f"[{self.timestamp}] Task: {self.task_summary} | Result: {self.result_summary}"


# ══════════════════════════════════════════════════════════════════════════════
# Keyword RAG Engine (Replaces Vector Embeddings)
# ══════════════════════════════════════════════════════════════════════════════


STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom",
])


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    text = text.lower().strip()
    words = re.findall(r'[a-z0-9_]+', text)
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return keywords


def keyword_overlap_score(query_keywords: list[str], entry_keywords: list[str]) -> float:
    """Score relevance by keyword overlap using TF-IDF-like weighting."""
    if not query_keywords or not entry_keywords:
        return 0.0
    query_set = set(query_keywords)
    entry_counter = Counter(entry_keywords)
    overlap = query_set & set(entry_keywords)
    if not overlap:
        return 0.0
    # Weight by how many query terms match and their frequency in the entry
    score = sum(1 + math.log(1 + entry_counter[kw]) for kw in overlap)
    # Normalize by query size
    return score / len(query_set)


# ══════════════════════════════════════════════════════════════════════════════
# Adaptation RAG (Keyword-Based Rule Retrieval)
# ══════════════════════════════════════════════════════════════════════════════


def retrieve_relevant_rules(rules: list[str], prompt: str, max_rules: int = 10) -> list[str]:
    """Retrieve only the adaptation rules relevant to the current prompt.
    Uses keyword matching to avoid bloating context with unrelated rules."""
    if not rules:
        return []

    prompt_keywords = extract_keywords(prompt)
    if not prompt_keywords:
        return rules[:max_rules]  # Fallback: return first N rules

    scored = []
    for rule in rules:
        rule_keywords = extract_keywords(rule)
        score = keyword_overlap_score(prompt_keywords, rule_keywords)
        scored.append((score, rule))

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return rules with any overlap, up to max_rules
    relevant = [rule for score, rule in scored if score > 0][:max_rules]

    # If no rules match at all, don't return any (they're not relevant)
    return relevant


# ══════════════════════════════════════════════════════════════════════════════
# Memory Store
# ══════════════════════════════════════════════════════════════════════════════


class MemoryStore:
    """Per-profile persistent memory with keyword-based RAG search."""

    def __init__(self, profile: str, config: Optional[MemoryConfig] = None):
        self.profile = profile
        self.config = config or MemoryConfig.from_env()
        self.entries: list[MemoryEntry] = []
        self.summaries: list[str] = []
        self._pending_summarization = False
        self._memory_file = Path(self.config.memory_file)
        self._load()

    def _ensure_dir(self):
        if self._memory_file.parent:
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self._memory_file.exists():
            return
        try:
            full_data = json.loads(self._memory_file.read_text())
            data = full_data.get(self.profile, {})
            if data:
                self.entries = [MemoryEntry.from_dict(e) for e in data.get("entries", [])]
                self.summaries = data.get("summaries", [])
        except Exception as exc:
            logger.error("Failed to load memory for '%s': %s", self.profile, exc)

    def _save(self):
        try:
            self._ensure_dir()
            full_data = {}
            if self._memory_file.exists():
                try:
                    full_data = json.loads(self._memory_file.read_text())
                except Exception:
                    pass

            full_data[self.profile] = {
                "entries": [e.to_dict() for e in self.entries],
                "summaries": self.summaries,
            }
            self._memory_file.write_text(json.dumps(full_data, indent=2))
        except Exception as exc:
            logger.error("Failed to save memory: %s", exc)

    @property
    def needs_summarization(self) -> bool:
        return len(self.entries) >= self.config.max_entries

    def add(self, task: str, result: str, tags: Optional[list[str]] = None):
        if not self.config.enabled:
            return

        # Extract keywords for RAG search
        combined_text = f"{task} {result}"
        keywords = extract_keywords(combined_text)

        entry = MemoryEntry(
            id=hashlib.md5(
                f"{task}{result}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            task_summary=task[:200],
            result_summary=result[:500],
            tags=tags or [],
            keywords=keywords[:30],  # Keep top 30 keywords per entry
        )
        self.entries.append(entry)
        if self.needs_summarization:
            self._pending_summarization = True
        self._save()

    def request_summarization(self) -> str:
        if not self._pending_summarization:
            return "No summarization needed."
        if len(self.entries) <= self.config.keep_after_summary:
            self._pending_summarization = False
            return "Not enough entries to summarize."

        to_summarize = self.entries[:-self.config.keep_after_summary]
        self.entries = self.entries[-self.config.keep_after_summary:]

        parts = [f"- {e.task_summary}: {e.result_summary[:100]}" for e in to_summarize]
        summary = (
            f"[Summary of {len(to_summarize)} tasks from "
            f"{to_summarize[0].timestamp} to {to_summarize[-1].timestamp}]:\n"
            + "\n".join(parts[:20])
        )
        if len(parts) > 20:
            summary += f"\n...and {len(parts) - 20} more tasks."

        self.summaries.append(summary)
        if len(self.summaries) > 5:
            self.summaries = self.summaries[-5:]

        self._pending_summarization = False
        self._save()
        return f"Summarized {len(to_summarize)} entries. {len(self.entries)} remain."

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """RAG search: find entries by keyword overlap."""
        if not self.entries:
            return []
        query_keywords = extract_keywords(query)
        if not query_keywords:
            return self.entries[-top_k:]  # Fallback: return most recent

        scored = []
        for entry in self.entries:
            score = keyword_overlap_score(query_keywords, entry.keywords)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def get_context(self, current_task: str, max_chars: int = 2000) -> str:
        if not self.config.enabled:
            return ""
        parts = []
        if self.summaries:
            parts.append("=== Historical Context ===")
            for summary in self.summaries[-2:]:
                parts.append(summary)
        relevant = self.search(current_task, top_k=5)
        if relevant:
            parts.append("\n=== Relevant Past Tasks ===")
            for entry in relevant:
                parts.append(entry.to_text())
        context = "\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[...truncated...]"
        return context

    def get_all(self) -> list[MemoryEntry]:
        return self.entries.copy()

    def list_entries(self) -> list[MemoryEntry]:
        return self.entries.copy()

    def clear(self):
        self.reset()

    def reset(self):
        self.entries = []
        self.summaries = []
        self._pending_summarization = False
        self._save()

    def stats(self) -> dict:
        return {
            "profile": self.profile,
            "entries": len(self.entries),
            "summaries": len(self.summaries),
            "enabled": self.config.enabled,
            "max_entries": self.config.max_entries,
            "pending_summarization": self._pending_summarization,
        }


_memory_stores: dict[str, MemoryStore] = {}


def get_memory(profile: str = "default") -> MemoryStore:
    if profile not in _memory_stores:
        _memory_stores[profile] = MemoryStore(profile)
    return _memory_stores[profile]


def reset_memory(profile: str = "default"):
    get_memory(profile).reset()


# ══════════════════════════════════════════════════════════════════════════════
# Feedback Store (for RLHF)
# ══════════════════════════════════════════════════════════════════════════════


class FeedbackStore:
    """Stores user feedback on agent responses for RLHF fine-tuning data generation."""

    def __init__(self, feedback_file: str = "data/feedback.json"):
        self._file = Path(feedback_file)
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                self._entries = json.loads(self._file.read_text())
            except Exception:
                self._entries = []

    def _save(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._entries, indent=2))
        except Exception as exc:
            logger.error("Failed to save feedback: %s", exc)

    def add(self, prompt: str, response: str, rating: str, message_id: str = ""):
        entry = {
            "id": message_id or hashlib.md5(f"{prompt}{response}{time.time()}".encode()).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "rating": rating,  # "positive" or "negative"
        }
        self._entries.append(entry)
        self._save()
        return entry["id"]

    def export_jsonl(self, output_path: str = "data/rlhf_dataset.jsonl") -> str:
        """Export feedback as JSONL for RLHF fine-tuning."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for entry in self._entries:
                jsonl_entry = {
                    "prompt": entry["prompt"],
                    "chosen" if entry["rating"] == "positive" else "rejected": entry["response"],
                    "rating": entry["rating"],
                    "timestamp": entry["timestamp"],
                }
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                count += 1
        return f"Exported {count} entries to {out}"

    def get_all(self) -> list[dict]:
        return self._entries.copy()

    def count(self) -> int:
        return len(self._entries)


_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store(feedback_file: str = "data/feedback.json") -> FeedbackStore:
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore(feedback_file)
    return _feedback_store
