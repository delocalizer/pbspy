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
import logging
import sys
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from os import getcwd, environ, path
from typing import Optional, Sequence

from asyncio_throttle import Throttler
from returns.future import future_safe, FutureResultE
from returns.io import IOFailure, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Success

LOGFORMAT = FORMAT = '%(asctime)s %(levelname)-6s %(name)-12s %(message)s'
LOGLEVEL = environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL, format=LOGFORMAT)
LOGGER = logging.getLogger(__name__)

# Some PBS installations may be configured differently
DEFAULT_ERR_PATH: str = '{cwd}/{jobname}.e{jobnum}'
# 10 jobs every 5 seconds
SUBMISSION_LIMIT: Throttler = Throttler(rate_limit=10, period=5)


@dataclass(kw_only=True)
class JobSpec:
    """Specifies a PBS batch job.

    Specifications not managed by explicit attributes may be supplied as-is
    via `extras` e.g. `extras='-l ngpus=1'`.

    IMPORTANT: `qsub` options that prevent a local error file being created
    (e.g. `-R e`) must not be used. The local error file is used to detect
    when the job is completed. By default the error file will be looked for
    in the current working directory; a custom location may be specified
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

    def __str__(self) -> str:
        """`qsub` command required to submit the job.

        `cmd` is not included and is assumed to be passed via stdin.
        """
        return (
            f'qsub'
            f' -l mem="{self.mem}"'
            f' -l ncpus="{self.ncpus}"'
            f' -l walltime="{self.walltime}"'
            f' -N "{self.name}"'
            f'{(" -q " + self.queue) if self.queue else ""}'
            f'{(" " + self.extras) if self.extras else ""}'
            f'{(" -e " + self.error_path) if self.error_path else ""}'
        )


@dataclass
class Job:
    """Describes a submitted PBS job."""

    jobid: str
    name: str
    complete_on_file: str

    def __str__(self) -> str:
        """Human-friendly representation"""
        return f'{self.name} [{self.jobid}]'

    def poll(self) -> str:
        """command required to poll the job."""
        return f'qstat -f -x -F json {self.jobid}'


def run(jobs: Sequence[JobSpec]) -> None:
    """Submit and monitor PBS jobs till they're all done."""
    asyncio.run(_run(jobs))


async def _run(jobs: Sequence[JobSpec]) -> None:
    """Compose the job submission, polling and result handling"""
    await asyncio.gather(
        *[flow(job, _submit, bind(_wait_till_done), _handle) for job in jobs]
    )


@future_safe
async def _submit(jobspec: JobSpec) -> Job:
    """Submit a job to the scheduler."""
    async with SUBMISSION_LIMIT:
        proc = await asyncio.create_subprocess_shell(
            str(jobspec), stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
        stdout, stderr = await proc.communicate(input=jobspec.cmd.encode('utf-8'))
        if proc.returncode != 0:
            raise RuntimeError(f'{jobspec}: {render(stderr)}')
        jobid = render(stdout)
        error_path = jobspec.error_path or DEFAULT_ERR_PATH.format(
            cwd=getcwd(), jobname=jobspec.name, jobnum=int(jobid.split('.')[0])
        )
        job = Job(jobid, jobspec.name, error_path)
        LOGGER.info('Submitted %s', job)
        return job


@future_safe
async def _wait_till_done(
        job: Job, interval: int = 10,
        timeout: int = 24*60*60) -> Job:
    """Wait until the job is finished.

    Args:
        interval: `int` seconds to wait between polling
        timeout: `int` seconds to wait before giving up
    """
    try:
        async with asyncio.timeout(timeout):
            while True:
                await asyncio.sleep(interval)
                LOGGER.debug('Checking %s', job)
                # filesystem checks are cheap but qstat is not, so we only run
                # qstat once: after we detect the file that signals completion.
                if not path.exists(job.complete_on_file):
                    continue
                proc = await asyncio.create_subprocess_shell(
                    job.poll(), stdout=PIPE, stderr=PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f'{job}: {render(stderr)}')
                details = json.loads(render(stdout))['Jobs'][job.jobid]
                exit_status = details['Exit_status']
                if exit_status != 0:
                    raise RuntimeError(f'{job}: {details["comment"]}')
                return job
    except asyncio.TimeoutError:
        raise RuntimeError(f'{job}: timed out after {timeout}s')


async def _handle(result: FutureResultE[Job]) -> None:
    """Do something with the pipeline result."""
    match (await result):
        case IOFailure(ex):
            LOGGER.error(ex)
        case IOSuccess(Success(job)):
            LOGGER.info('%s succeeded', job.jobid)


def render(byts: bytes, encoding='utf-8') -> str:
    """Decode bytes and strip whitespace"""
    return byts.decode(encoding).strip()


if __name__ == '__main__':
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
            name='the-aged-aged-man',
            extras='-l chip=Intel',
        ),
    ]
    run(jerbs)
