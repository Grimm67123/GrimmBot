"""
GrimmBot — Persistent Task Scheduler
One-time, daily, and interval scheduling with disk persistence.
"""

import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger("scheduler")

try:
    import pytz
except ImportError:
    import datetime as _dt
    class _MockPytz:
        UTC = _dt.timezone.utc
        @staticmethod
        def timezone(zone):
            return _dt.timezone.utc
    pytz = _MockPytz()


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SchedulerConfig:
    persistent: bool = True
    check_interval: int = 30
    data_dir: str = "data/scheduler"
    timezone: str = "UTC"

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        return cls(
            persistent=os.getenv("SCHEDULER_PERSISTENT", "true").lower() == "true",
            check_interval=int(os.getenv("SCHEDULER_CHECK_INTERVAL", "30")),
            data_dir=os.getenv("SCHEDULER_DIR", "data/scheduler"),
            timezone=os.getenv("TIMEZONE", "UTC"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Data Model
# ══════════════════════════════════════════════════════════════════════════════


class ScheduleType(Enum):
    ONCE = "once"
    DAILY = "daily"
    INTERVAL = "interval"


@dataclass
class ScheduledTask:
    id: str
    prompt: str
    schedule_type: str
    schedule_value: str
    profile: str = "default"
    created_at: str = ""
    last_run: Optional[str] = None
    run_count: int = 0
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledTask":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════


class PersistentScheduler:
    """Background scheduler with disk persistence."""

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig.from_env()
        self.tasks: dict[str, ScheduledTask] = {}
        self.tz = pytz.timezone(self.config.timezone)
        self._callbacks: list[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._data_file = Path(self.config.data_dir) / "scheduled_tasks.json"
        self._ensure_dir()
        self._load()

    def _ensure_dir(self):
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self.config.persistent or not self._data_file.exists():
            return
        try:
            data = json.loads(self._data_file.read_text())
            for td in data.get("tasks", []):
                task = ScheduledTask.from_dict(td)
                self.tasks[task.id] = task
        except Exception as exc:
            logger.error("Failed to load scheduled tasks: %s", exc)

    def _save(self):
        if not self.config.persistent:
            return
        try:
            data = {"tasks": [t.to_dict() for t in self.tasks.values()]}
            self._data_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.error("Failed to save scheduled tasks: %s", exc)

    def add_callback(self, callback: Callable[[ScheduledTask], None]):
        self._callbacks.append(callback)

    def schedule_once(self, prompt: str, run_at: datetime,
                      profile: str = "default", tags: Optional[list[str]] = None) -> str:
        task_id = f"once_{int(run_at.timestamp())}"
        task = ScheduledTask(
            id=task_id, prompt=prompt,
            schedule_type=ScheduleType.ONCE.value,
            schedule_value=run_at.isoformat(),
            profile=profile,
            created_at=datetime.now(self.tz).isoformat(),
            tags=tags or [],
        )
        with self._lock:
            self.tasks[task_id] = task
            self._save()
        return task_id

    def schedule_daily(self, prompt: str, hour: int, minute: int = 0,
                       profile: str = "default", tags: Optional[list[str]] = None) -> str:
        task_id = f"daily_{hour:02d}{minute:02d}_{hashlib.md5(prompt.encode()).hexdigest()[:6]}"
        task = ScheduledTask(
            id=task_id, prompt=prompt,
            schedule_type=ScheduleType.DAILY.value,
            schedule_value=f"{hour:02d}:{minute:02d}",
            profile=profile,
            created_at=datetime.now(self.tz).isoformat(),
            tags=tags or [],
        )
        with self._lock:
            self.tasks[task_id] = task
            self._save()
        return task_id

    def schedule_interval(self, prompt: str, minutes: int,
                          profile: str = "default", tags: Optional[list[str]] = None) -> str:
        task_id = f"interval_{minutes}m_{hashlib.md5(prompt.encode()).hexdigest()[:6]}"
        task = ScheduledTask(
            id=task_id, prompt=prompt,
            schedule_type=ScheduleType.INTERVAL.value,
            schedule_value=str(minutes),
            profile=profile,
            created_at=datetime.now(self.tz).isoformat(),
            tags=tags or [],
        )
        with self._lock:
            self.tasks[task_id] = task
            self._save()
        return task_id

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._save()
                return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self.tasks.get(task_id)

    def _is_due(self, task: ScheduledTask, now: datetime) -> bool:
        if not task.enabled:
            return False
        if task.schedule_type == ScheduleType.ONCE.value:
            target = datetime.fromisoformat(task.schedule_value)
            if target.tzinfo is None:
                target = self.tz.localize(target)
            return now >= target
        elif task.schedule_type == ScheduleType.DAILY.value:
            hour, minute = map(int, task.schedule_value.split(":"))
            target_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if task.last_run:
                last = datetime.fromisoformat(task.last_run)
                if last.tzinfo is None:
                    last = self.tz.localize(last)
                if last.date() == now.date():
                    return False
            return now >= target_today
        elif task.schedule_type == ScheduleType.INTERVAL.value:
            interval_minutes = int(task.schedule_value)
            if not task.last_run:
                return True
            last = datetime.fromisoformat(task.last_run)
            if last.tzinfo is None:
                last = self.tz.localize(last)
            return now >= last + timedelta(minutes=interval_minutes)
        return False

    def _check_and_run(self):
        now = datetime.now(self.tz)
        due_tasks = []
        with self._lock:
            for task in self.tasks.values():
                if self._is_due(task, now):
                    due_tasks.append(task)
            for task in due_tasks:
                task.last_run = now.isoformat()
                task.run_count += 1
                if task.schedule_type == ScheduleType.ONCE.value:
                    del self.tasks[task.id]
            if due_tasks:
                self._save()
        for task in due_tasks:
            for callback in self._callbacks:
                try:
                    callback(task)
                except Exception as exc:
                    logger.error("Callback error for task '%s': %s", task.id, exc)

    def _run_loop(self):
        while self._running:
            try:
                self._check_and_run()
            except Exception as exc:
                logger.error("Scheduler loop error: %s", exc)
            time.sleep(self.config.check_interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="scheduler")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def stats(self) -> dict:
        return {
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
            "running": self._running,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_scheduler: Optional[PersistentScheduler] = None


def get_scheduler() -> PersistentScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PersistentScheduler()
    return _scheduler
