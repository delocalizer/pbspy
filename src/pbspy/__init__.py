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
from typing import Optional, Sequence, Tuple

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

# 72 hr timeout
DEFAULT_TIMEOUT: int = 3*24*60*60
# 10 jobs every 5 seconds
SUBMISSION_LIMIT: Throttler = Throttler(rate_limit=10, period=5)


# Two soft design goals that are somewhat in tension:
# Goal 1:
#   Avoid data structures with rich behavior...
# Goal 2: 
#   ... but keep all behavior that depends on details of how the scheduler
#   is configured, and possibly even which scheduler is used, out of the
#   module-level functions. 

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
    name: str
    queue: Optional[str] = None
    extras: Optional[str] = None
    error_path: Optional[str] = None

    @property
    def cmd_b(self) -> bytes:
        """`self.cmd` encoded as bytes"""
        return self.cmd.encode('utf-8')

    def __str__(self) -> str:
        """Representation as the `qsub` command for submitting the job.

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


@dataclass(kw_only=True)
class Job:
    """Describes a submitted PBS job."""

    jobid: str
    name: str
    complete_on_file: str

    @property
    def jobnum(self) -> int:
        """The integer part of `self.jobid`"""
        return int(self.jobid.split('.')[0])

    def __post_init__(self):
        """Set default for `self.complete_on_file` if initialized to None."""
        if not self.complete_on_file:
            self.complete_on_file = path.join(
                # some PBS installations may be configured differently
                getcwd(), f'{self.name}.e{self.jobnum}')

    def __str__(self) -> str:
        """Human-friendly representation"""
        return f'{self.name} [{self.jobid}]'

    def status_cmd(self) -> str:
        """Command to check status of finished job."""
        return f'qstat -f -x -F json {self.jobid}'

    def status_check(self, output: bytes) -> Tuple[bool, str]:
        """Parse the output of `status_cmd`.

        Returns:
            A tuple (success, msg) where success is True iff job exit status
            was 0, and msg is any helpful info to use e.g. when success==False.
        """
        details = json.loads(decode_(output))['Jobs'][self.jobid]
        return (details['Exit_status'] == 0, details['comment'])


def run(jobs: Sequence[JobSpec]):
    """Submit and monitor PBS jobs till they're all done."""
    asyncio.run(_run(jobs))


async def _run(jobs: Sequence[JobSpec]):
    """Compose the job submission, polling and result handling"""
    await asyncio.gather(
        *[flow(job, submit, bind(wait_till_done), handle) for job in jobs]
    )


@future_safe
async def submit(jobspec: JobSpec) -> Job:
    """Submit a job to the scheduler."""
    async with SUBMISSION_LIMIT:
        proc = await asyncio.create_subprocess_shell(
            str(jobspec), stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
        stdout, stderr = await proc.communicate(input=jobspec.cmd_b)
        if proc.returncode != 0:
            raise RuntimeError(f'{jobspec}: {decode_(stderr)}')
        job = Job(jobid=decode_(stdout),
                  name=jobspec.name,
                  complete_on_file=jobspec.error_path)
        LOGGER.info('Submitted %s', job)
        return job


@future_safe
async def wait_till_done(
        job: Job, interval: int = 10,
        timeout: int = DEFAULT_TIMEOUT) -> Job:
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
                    job.status_cmd(), stdout=PIPE, stderr=PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f'{job}: {decode_(stderr)}')
                success, msg = job.status_check(stdout)
                if not success:
                    raise RuntimeError(f'{job}: {msg}')
                return job
    except asyncio.TimeoutError:
        raise RuntimeError(f'{job}: timed out after {timeout}s')


async def handle(result: FutureResultE[Job]):
    """Do something with the pipeline result."""
    match (await result):
        case IOFailure(ex):
            LOGGER.error(ex)
        case IOSuccess(Success(job)):
            LOGGER.info('%s succeeded', job.jobid)


def decode_(byts: bytes, encoding='utf-8') -> str:
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
