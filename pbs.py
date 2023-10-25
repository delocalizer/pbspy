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
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from os import getcwd, path
from sys import stderr
from typing import Any, Dict, Optional, Sequence

from returns.future import future_safe
from returns.io import IOFailure, IOResult, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Success


@dataclass
class JobSpec:
    """Specifies a PBS batch job."""

    cmd: str
    name: str
    ask_mem: str
    ask_ncpus: int
    ask_walltime: str
    path_err: Optional[str] = None
    path_out: Optional[str] = None


@dataclass
class Job:
    """Describes a submitted PBS job."""

    job_id: int
    err_path: str


@future_safe
async def submit(jobspec: Optional[JobSpec | Dict[str, Any]] = None, **kwargs) -> Job:
    """Submit a job to the scheduler."""
    match jobspec:
        case JobSpec():
            pass
        case dict():
            jobspec = JobSpec(**jobspec)
        case _:
            jobspec = JobSpec(**kwargs)

    proc = await asyncio.create_subprocess_shell(
        f'''
        qsub \\
        -l mem={jobspec.ask_mem} \\
        -l ncpus={jobspec.ask_ncpus} \\
        -l walltime={jobspec.ask_walltime} \\
        -N "{jobspec.name}"''',
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = await proc.communicate(input=jobspec.cmd.encode("utf-8"))
    if proc.returncode != 0:
        raise Exception(f'{jobspec}: {stderr.decode("utf-8")}')
    job_id = int(stdout.decode("utf-8").split(".")[0].strip())
    job_name = jobspec.name or "STDIN"
    err_path = jobspec.path_err or f"{getcwd()}/{job_name}.e{job_id}"
    return Job(job_id, err_path)


@future_safe
async def wait_till_done(job: Job, waitsec=10) -> Job:
    """Wait until the job is finished."""
    while True:
        if path.exists(job.err_path):
            # TODO: single qstat call to check job exit status
            return job 
        await asyncio.sleep(waitsec)


def handle_results(results: Sequence[IOResult[Job, Exception]]) -> None:
    """Do something with accumulated results."""
    for result in results:
        match result:
            case IOFailure(ex):
                print(ex, file=stderr)
            case IOSuccess(Success(job)):
                print(f'{job.job_id} completed')


async def main():
    jobs = [
        JobSpec(
            cmd="echo foo",
            name="a sitting-on-a-gate",
            ask_mem="1MB",
            ask_ncpus=1,
            ask_walltime="00:01:00",
        ),
        JobSpec(
            cmd="echo bar",
            name="haddocks-eyes",
            ask_mem="1MB",
            ask_ncpus=1,
            ask_walltime="00:01:00",
        ),
    ]

    # TODO: Semaphore to throttle number of job submissions and checks
    flows = [flow(job, submit, bind(wait_till_done)) for job in jobs]
    handle_results(await asyncio.gather(*flows))


if __name__ == "__main__":
    asyncio.run(main())
