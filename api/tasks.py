"""
DriftInsights — Background Tasks (Thread-based)

Provides async-like task execution using threads.
Handles retraining, SHAP computation, and metrics aggregation.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

logger = logging.getLogger("driftinsights.tasks")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    task_id: str
    task_name: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "duration": self.duration,
            "error": self.error,
        }


class TaskManager:
    """
    Thread-based background task manager.

    Provides a simple interface for running CPU-bound tasks
    (retraining, SHAP computation) in background threads.

    Usage:
        tasks = TaskManager(max_workers=2)
        task_id = tasks.submit("retrain", retrain_function, arg1, arg2)
        result = tasks.get_result(task_id)
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskResult] = {}
        self._futures: dict[str, Future] = {}
        self._counter = 0
        self._lock = threading.Lock()

        logger.info(f"TaskManager initialized with {max_workers} workers")

    def submit(self, task_name: str, fn: Callable, *args, **kwargs) -> str:
        """
        Submit a task for background execution.

        Args:
            task_name: Human-readable task name
            fn: Callable to execute
            *args, **kwargs: Arguments passed to fn

        Returns:
            task_id: Unique identifier for tracking
        """
        with self._lock:
            self._counter += 1
            task_id = f"task_{self._counter}_{task_name}"

        task_result = TaskResult(
            task_id=task_id,
            task_name=task_name,
            status=TaskStatus.PENDING,
        )
        self._tasks[task_id] = task_result

        def _wrapped():
            task_result.status = TaskStatus.RUNNING
            task_result.start_time = time.time()
            try:
                result = fn(*args, **kwargs)
                task_result.result = result
                task_result.status = TaskStatus.COMPLETED
                logger.info(f"Task {task_id} completed")
            except Exception as e:
                task_result.error = str(e)
                task_result.status = TaskStatus.FAILED
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                task_result.end_time = time.time()

        future = self._executor.submit(_wrapped)
        self._futures[task_id] = future

        logger.info(f"Task submitted: {task_id}")
        return task_id

    def get_result(self, task_id: str, timeout: float | None = None) -> TaskResult:
        """Get the result of a task, optionally waiting for completion."""
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")

        if timeout and task_id in self._futures:
            self._futures[task_id].result(timeout=timeout)

        return self._tasks[task_id]

    def is_running(self, task_name: str) -> bool:
        """Check if any task with the given name is currently running."""
        for task in self._tasks.values():
            if task.task_name == task_name and task.status == TaskStatus.RUNNING:
                return True
        return False

    def get_all_tasks(self) -> list[dict]:
        """Get status of all tasks."""
        return [t.to_dict() for t in self._tasks.values()]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)
        logger.info("TaskManager shutdown")


# Global task manager
task_manager = TaskManager(max_workers=2)
