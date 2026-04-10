"""
Tests for scheduler.py — PersistentScheduler, ScheduledTask.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from memory import (
    SchedulerConfig, ScheduleType, ScheduledTask,
    PersistentScheduler, get_scheduler,
)


class TestScheduledTask:
    def test_to_dict_from_dict(self):
        task = ScheduledTask(
            id="test_1", prompt="Do something",
            schedule_type="once", schedule_value="2026-01-01T00:00:00",
            profile="default", created_at="2026-01-01T00:00:00",
        )
        d = task.to_dict()
        assert d["id"] == "test_1"
        restored = ScheduledTask.from_dict(d)
        assert restored.id == "test_1"
        assert restored.prompt == "Do something"


class TestPersistentScheduler:
    def test_schedule_once(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"), persistent=True)
        sched = PersistentScheduler(cfg)
        target = datetime.now(timezone.utc) + timedelta(hours=1)
        task_id = sched.schedule_once("Test task", target)
        assert task_id.startswith("once_")
        assert len(sched.list_tasks()) == 1

    def test_schedule_daily(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"), persistent=True)
        sched = PersistentScheduler(cfg)
        task_id = sched.schedule_daily("Daily check", 9, 30)
        assert task_id.startswith("daily_")
        task = sched.get_task(task_id)
        assert task.schedule_value == "09:30"

    def test_schedule_interval(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"), persistent=True)
        sched = PersistentScheduler(cfg)
        task_id = sched.schedule_interval("Repeat task", 15)
        assert task_id.startswith("interval_")
        task = sched.get_task(task_id)
        assert task.schedule_value == "15"

    def test_cancel_task(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"), persistent=True)
        sched = PersistentScheduler(cfg)
        target = datetime.now(timezone.utc) + timedelta(hours=1)
        task_id = sched.schedule_once("Cancel me", target)
        assert sched.cancel(task_id) is True
        assert len(sched.list_tasks()) == 0
        assert sched.cancel("nonexistent") is False

    def test_persistence(self, tmp_path):
        sched_dir = str(tmp_path / "sched")
        cfg = SchedulerConfig(data_dir=sched_dir, persistent=True)
        sched1 = PersistentScheduler(cfg)
        sched1.schedule_daily("Persistent task", 12, 0)

        sched2 = PersistentScheduler(cfg)
        assert len(sched2.list_tasks()) == 1
        assert sched2.list_tasks()[0].prompt == "Persistent task"

    def test_is_due_once(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"))
        sched = PersistentScheduler(cfg)
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        task = ScheduledTask(
            id="test", prompt="past", schedule_type="once",
            schedule_value=past.isoformat(),
        )
        assert sched._is_due(task, datetime.now(timezone.utc)) is True

    def test_is_due_disabled(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"))
        sched = PersistentScheduler(cfg)
        task = ScheduledTask(
            id="test", prompt="disabled", schedule_type="once",
            schedule_value=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            enabled=False,
        )
        assert sched._is_due(task, datetime.now(timezone.utc)) is False

    def test_callback_fired(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"))
        sched = PersistentScheduler(cfg)
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        sched.schedule_once("Fire me", past)

        fired = []
        sched.add_callback(lambda task: fired.append(task.prompt))
        sched._check_and_run()

        assert "Fire me" in fired
        assert len(sched.list_tasks()) == 0

    def test_stats(self, tmp_path):
        cfg = SchedulerConfig(data_dir=str(tmp_path / "sched"))
        sched = PersistentScheduler(cfg)
        sched.schedule_daily("task", 10)
        stats = sched.stats()
        assert stats["total_tasks"] == 1
        assert stats["enabled_tasks"] == 1
