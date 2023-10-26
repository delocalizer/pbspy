"""
Lightweight PBS job submission and monitoring.

By using an `asyncio` event loop we can submit and monitor hundreds of jobs
(or more) without blocking on IO or using multiple threads. By using `returns`
we can use FP style composition and not worry about checking for failures at
every point.

References:

* https://docs.python.org/3/library/asyncio-task.html
* https://github.com/dry-python/returns#better-async-composition
"""
import asyncio
import json
import sys
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from os import getcwd, path
from typing import Any, Dict, List, Optional, Sequence

from asyncio_throttle import Throttler
from returns.future import future_safe, FutureResult
from returns.io import IOFailure, IOResult, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Success


# 10 jobs every 5 seconds
SUBMISSION_LIMIT: Throttler = Throttler(rate_limit=10, period=5)

# Some PBS installations may be configured differently
DEFAULT_ERR_PATH: str = '{cwd}/{jobname}.e{jobnum}'


@dataclass(kw_only=True)
class JobSpec:
    """Specifies a PBS batch job.

    Specifications that are not covered by explicit attributes may be
    supplied as-is via `extras` e.g. `extras='-l ngpus=1'`.

    IMPORTANT: options that prevent a local error file being created
    (e.g. `-R e`) must not be used. The local error file is used to detect
    when the job is completed. By default the error file will be looked for
    in the current working directory but a custom location may be specified
    using `error_path`.
    """

    cmd: str
    mem: str
    ncpus: int
    walltime: str
    name: str = 'pbspy'
    queue: Optional[str] = None
    extras: Optional[str] = None
    error_path: Optional[str] = None

    def __repr__(self) -> str:
        """`qsub` command required to submit the job.

        `cmd` is not included and is assumed to be passed via stdin.
        """
        return (f'qsub'
                f' -l mem={self.mem}'
                f' -l ncpus={self.ncpus}'
                f' -l walltime={self.walltime}'
                f' -N {self.name}'
                f'{(" -q " + self.queue) if self.queue else ""}'
                f'{(" -e " + self.error_path) if self.error_path else ""}'
                f' {self.extras}')


@dataclass
class Job:
    """Describes a submitted PBS job."""

    jobid: str
    complete_on_file: str

    def poll(self) -> str:
        """command required to poll the job."""
        return f'qstat -f -x -F json {self.jobid}'


@future_safe
async def submit(jobspec: JobSpec) -> Job:
    """Submit a job to the scheduler."""
    async with SUBMISSION_LIMIT:
        proc = await asyncio.create_subprocess_shell(
            repr(jobspec),
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await proc.communicate(
            input=jobspec.cmd.encode("utf-8"))
        if proc.returncode != 0:
            raise Exception(f'{jobspec}: {stderr.decode("utf-8")}')
        jobid = stdout.decode("utf-8").strip()
        error_path = jobspec.error_path or DEFAULT_ERR_PATH.format(
            cwd=getcwd(),
            jobname=jobspec.name,
            jobnum=int(jobid.split(".")[0]))
        print(f'{jobid} submitted')
        return Job(jobid, error_path)


@future_safe
async def wait_till_done(job: Job, waitsec: int = 10) -> Job:
    """Wait until the job is finished."""
    while True:
        if path.exists(job.complete_on_file):
            # filesystem checks are cheap but qstat is not, so we only qstat
            # once: after we detect the file that signals completion.
            proc = await asyncio.create_subprocess_shell(
                job.poll(),
                stdout=PIPE,
                stderr=PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise Exception(f'{job}: {stderr.decode("utf-8")}')
            details = json.loads(stdout.decode('utf-8'))['Jobs'][job.jobid]
            exit_status = details['Exit_status']
            if exit_status != 0:
                raise Exception(f'{job}: {details["comment"]}')
            return job
        await asyncio.sleep(waitsec)


async def run(jobs: Sequence[JobSpec]) -> None:
    """Submit and monitor PBS jobs till they're all done."""
    flows: List[FutureResult] = [
            flow(job, submit, bind(wait_till_done)) for job in jobs]
    handle_results(await asyncio.gather(*flows))


def handle_results(results: Sequence[IOResult[Job, Exception]]) -> None:
    """Do something with accumulated results."""
    for result in results:
        match result:
            case IOFailure(ex):
                print(ex, file=sys.stderr)
            case IOSuccess(Success(job)):
                print(f'{job.jobid} succeeded')


if __name__ == "__main__":
    # demo
    jerbs = [
        JobSpec(
            cmd='echo foo',
            mem='1MB',
            ncpus=1,
            walltime='00:01:00',
            name='a-sitting-on-a-gate',
            queue='testing',
        ),
        JobSpec(
            cmd='echo bar',
            mem='1MB',
            ncpus=1,
            walltime='00:01:00',
            name='haddocks-eyes',
        ),
    ]
    asyncio.run(run(jerbs))
