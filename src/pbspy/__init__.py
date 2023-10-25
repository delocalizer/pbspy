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
from asyncio.subprocess import PIPE
from asyncio_throttle import Throttler
from dataclasses import dataclass
from os import getcwd, path
from sys import stderr
from typing import Any, Dict, Optional, Sequence

from returns.future import future_safe
from returns.io import IOFailure, IOResult, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Success


DEFAULT_JOBNAME = 'pbspy'
QSTAT_TEMPLATE = '''qstat -f -x -F json {jobid}'''
# 10 jobs every 5 seconds
SUBMISSION_LIMIT: Throttler = Throttler(rate_limit=10, period=5)


@dataclass
class JobSpec:
    """Specifies a PBS batch job."""

    cmd: str
    mem: str
    ncpus: int
    walltime: str
    name: str = DEFAULT_JOBNAME
    queue: Optional[str] = None
    error_path: Optional[str] = None
    output_path: Optional[str] = None

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
                f'{(" -o " + self.output_path) if self.output_path else ""}')


@dataclass
class Job:
    """Describes a submitted PBS job."""

    jobid: int
    error_path: str

    def poll(self) -> str:
        """command required to poll the job."""
        return f'qstat -f -x -F json {self.jobid}'


@future_safe
async def submit(jobspec: Optional[JobSpec | Dict[str, Any]] = None,
                 **kwargs) -> Job:
    """Submit a job to the scheduler."""
    match jobspec:
        case JobSpec():
            pass
        case dict():
            jobspec = JobSpec(**jobspec)
        case _:
            jobspec = JobSpec(**kwargs)

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
        jobnum = int(jobid.split(".")[0])
        error_path = jobspec.error_path or f"{getcwd()}/{jobspec.name}.e{jobnum}"
        print(f'{jobid} submitted')
        return Job(jobid, error_path)


@future_safe
async def wait_till_done(job: Job, waitsec=10) -> Job:
    """Wait until the job is finished."""
    # filesystem checks are cheap, qstat is not
    while True:
        if path.exists(job.error_path):
            proc = await asyncio.create_subprocess_shell(
                job.poll(),
                stdout=PIPE,
                stderr=PIPE
            )
            stdout, stderr = await proc.communicate()
            details = json.loads(stdout.decode('utf-8'))['Jobs'][job.jobid]
            exit_status = details['Exit_status']
            if exit_status != 0:
                raise Exception(f'{job}: {details["comment"]}')
            return job
        await asyncio.sleep(waitsec)


def handle_results(results: Sequence[IOResult[Job, Exception]]) -> None:
    """Do something with accumulated results."""
    for result in results:
        match result:
            case IOFailure(ex):
                print(ex, file=stderr)
            case IOSuccess(Success(job)):
                print(f'{job.jobid} succeeded')


async def main():
    jobs = [
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

    flows = [flow(job, submit, bind(wait_till_done)) for job in jobs]
    handle_results(await asyncio.gather(*flows))


if __name__ == "__main__":
    asyncio.run(main())
