"""The job worker.

Claims work from the queue, runs the matching pipeline stage, and decides what
a failure means. That decision is the whole point of the module:

* :class:`~thatone.errors.RetryableError` — back off and requeue. Rate
  limits, timeouts, 5xx.
* :class:`~thatone.errors.TerminalError` — record and move on. A refusal,
  a corrupt file, a malformed request. Retrying produces the identical failure
  and burns the budget doing it.
* :class:`~thatone.errors.AuthError` — stop the worker. Bad credentials
  fail every subsequent item, so continuing would march the entire queue into
  the failed state for a problem one environment variable would fix.

Attempts are capped, so an error miscategorised as retryable still terminates
rather than looping forever.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

from ..config import Settings
from ..errors import (
    AuthError,
    RetryableError,
    TerminalError,
    ThatOneError,
)
from ..indexer import Indexer
from ..models import Job, JobKind, MediaStatus, UsageRecord
from ..store.base import StorageBackend

log = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    claimed: int = 0
    completed: int = 0
    retried: int = 0
    failed: int = 0
    usage: UsageRecord = field(default_factory=UsageRecord)

    def __str__(self) -> str:  # pragma: no cover - logging aid
        return (
            f"claimed={self.claimed} completed={self.completed} "
            f"retried={self.retried} failed={self.failed}"
        )


class Worker:
    """Drains the job queue."""

    def __init__(self, settings: Settings, store: StorageBackend, indexer: Indexer) -> None:
        self.settings = settings
        self.store = store
        self.indexer = indexer
        self.stats = WorkerStats()
        self._stopping = False

    def stop(self) -> None:
        """Ask the worker to finish its current batch and exit."""
        self._stopping = True

    # -- execution ---------------------------------------------------------

    async def _execute(self, job: Job) -> None:
        if job.kind is JobKind.DESCRIBE:
            usage = await self.indexer.describe(job.media_id)
            self.stats.usage = self.stats.usage + usage
        elif job.kind is JobKind.EMBED:
            await self.indexer.embed(job.media_id)
        else:
            raise TerminalError(f"worker cannot run job kind {job.kind}")

    def _backoff(self, attempt: int, hinted: float | None) -> float:
        """Exponential backoff with jitter, honouring a provider hint.

        Jitter is not decoration: a batch run claims jobs in lockstep, so
        without it every worker retries at the same instant and re-triggers the
        rate limit that caused the backoff.
        """
        if hinted:
            return float(min(hinted, self.settings.jobs.backoff_max_seconds))
        base = self.settings.jobs.backoff_base_seconds * (2 ** (attempt - 1))
        capped = min(base, self.settings.jobs.backoff_max_seconds)
        return float(capped) * (0.5 + random.random())

    async def _handle_failure(self, job: Job, exc: Exception) -> None:
        assert job.id is not None
        max_attempts = self.settings.jobs.max_attempts

        if isinstance(exc, RetryableError) and job.attempts < max_attempts:
            delay = self._backoff(job.attempts, getattr(exc, "retry_after", None))
            await asyncio.to_thread(
                self.store.fail_job, job.id, str(exc), retry=True, backoff_seconds=delay
            )
            self.stats.retried += 1
            log.info(
                "retrying %s/%s in %.1fs (attempt %d/%d): %s",
                job.kind, job.media_id, delay, job.attempts, max_attempts, exc,
            )
            return

        # Terminal, or retryable but out of attempts. Either way the item stops
        # here and the reason is recorded on both the job and the media row, so
        # a failed item is explicable later without reading logs.
        reason = str(exc)
        if isinstance(exc, RetryableError):
            reason = f"gave up after {job.attempts} attempts: {reason}"
        await asyncio.to_thread(self.store.fail_job, job.id, reason, retry=False)
        await asyncio.to_thread(
            self.store.set_status, job.media_id, MediaStatus.FAILED, error=reason
        )
        self.stats.failed += 1
        log.warning("failed %s/%s: %s", job.kind, job.media_id, reason)

    async def _run_job(self, job: Job) -> None:
        assert job.id is not None
        try:
            await self._execute(job)
        except AuthError:
            # Never marks the item failed: the item is fine, the credentials
            # are not, and marking it failed would require re-ingesting a
            # perfectly good file once the key is fixed.
            raise
        except (ThatOneError, OSError) as exc:
            await self._handle_failure(job, exc)
        else:
            await asyncio.to_thread(self.store.complete_job, job.id)
            self.stats.completed += 1

    # -- batches -----------------------------------------------------------

    async def run_batch(self, kind: JobKind) -> int:
        """Claim and run one batch. Returns how many jobs ran."""
        jobs = await asyncio.to_thread(
            self.store.claim_jobs,
            kind,
            limit=self.settings.jobs.claim_batch_size,
            lease_seconds=self.settings.jobs.lease_seconds,
        )
        if not jobs:
            return 0
        self.stats.claimed += len(jobs)

        semaphore = asyncio.Semaphore(self.settings.jobs.concurrency)

        async def guarded(job: Job) -> None:
            async with semaphore:
                await self._run_job(job)

        # AuthError propagates out of the gather and stops the run; anything
        # else has already been recorded against its own job by _run_job.
        await asyncio.gather(*(guarded(job) for job in jobs))
        return len(jobs)

    async def drain(
        self, kinds: tuple[JobKind, ...] = (JobKind.DESCRIBE, JobKind.EMBED)
    ) -> WorkerStats:
        """Run until the queue is empty.

        Stages are drained in order because ``describe`` enqueues ``embed``;
        looping until a full pass does no work catches the jobs created by
        earlier passes.
        """
        await asyncio.to_thread(self.store.reclaim_expired_leases)
        while not self._stopping:
            done = 0
            for kind in kinds:
                done += await self.run_batch(kind)
            if done == 0:
                break
        return self.stats

    async def run_forever(self, *, poll_seconds: float = 2.0) -> None:
        """Poll the queue indefinitely.

        Expired leases are reclaimed on every idle pass, so work stranded by a
        killed worker returns to the pool without operator intervention.
        """
        while not self._stopping:
            reclaimed = await asyncio.to_thread(self.store.reclaim_expired_leases)
            if reclaimed:
                log.info("reclaimed %d expired lease(s)", reclaimed)

            done = 0
            for kind in (JobKind.DESCRIBE, JobKind.EMBED):
                done += await self.run_batch(kind)
            if done == 0:
                await asyncio.sleep(poll_seconds)
