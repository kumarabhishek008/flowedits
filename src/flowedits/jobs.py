"""Background jobs for long work (detection, rendering) so tool calls return before clients time out."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

from .workspace import ProjectError


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, project_id: str, kind: str):
        self.id = f"job-{uuid.uuid4().hex[:8]}"
        self.project_id = project_id
        self.kind = kind
        self.status = "running"
        self.progress = 0.0
        self.message = ""
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.result: Any = None
        self.preview: bytes | None = None
        self.error: str | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()

    def update(self, fraction: float, message: str | None = None) -> None:
        """Progress callback for the work function; raises Cancelled when cancel_job was called."""
        if self._cancel.is_set():
            raise Cancelled()
        self.progress = max(0.0, min(1.0, fraction))
        if message is not None:
            self.message = message

    def wait(self, seconds: float) -> bool:
        return self._done.wait(max(0.0, seconds))

    def cancel(self) -> None:
        self._cancel.set()

    def to_dict(self) -> dict:
        now = self.finished_at or time.time()
        elapsed = now - self.started_at
        d = {
            "job_id": self.id,
            "project_id": self.project_id,
            "kind": self.kind,
            "status": self.status,
            "progress_percent": round(self.progress * 100, 1),
            "elapsed_seconds": round(elapsed, 1),
        }
        if self.message:
            d["message"] = self.message
        if self.status == "running" and self.progress > 0.02:
            d["eta_seconds"] = round(elapsed / self.progress * (1 - self.progress))
        if self.error:
            d["error"] = self.error
        return d


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def running(self, project_id: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.project_id == project_id and j.status == "running"), None)

    def ensure_idle(self, project_id: str) -> None:
        job = self.running(project_id)
        if job:
            raise ProjectError(f"project {project_id} is busy with {job.kind} ({job.id}, "
                               f"{job.progress * 100:.0f}% done). Call job_status('{job.id}') and retry when it "
                               "finishes, or cancel_job.")

    def start(self, project_id: str, kind: str, fn: Callable[[Job], Any]) -> Job:
        with self._lock:
            busy = next((j for j in self._jobs.values() if j.project_id == project_id and j.status == "running"), None)
            if busy:
                raise ProjectError(f"project {project_id} is busy with {busy.kind} ({busy.id}); call "
                                   f"job_status('{busy.id}') first")
            job = Job(project_id, kind)
            self._jobs[job.id] = job

        def run() -> None:
            try:
                out = fn(job)
                if isinstance(out, tuple):
                    job.result, job.preview = out
                else:
                    job.result = out
                job.progress = 1.0
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:  # reported to the client through job_status
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)[-1500:]
            finally:
                job.finished_at = time.time()
                job._done.set()

        threading.Thread(target=run, name=job.id, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ProjectError(f"unknown job {job_id!r} (jobs are lost when the server restarts)")
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)]


JOBS = JobManager()
