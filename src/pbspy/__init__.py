"""Lightweight PBS job submission and monitoring."""

import asyncio
import json
import logging
import sys
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from os import getcwd, path
from typing import Optional, Sequence, Tuple

from asyncio_throttle import Throttler
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from returns.future import future_safe, FutureResultE
from returns.io import IOFailure, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Success


LOGGER = logging.getLogger(__name__)

# Technology choices:
#
# By using an `asyncio` event loop we can submit and monitor hundreds of jobs
# (or more) without blocking on IO or using multiple threads. By using
# `returns` we can use FP style composition and not worry about checking for
# failures at every point; for DI however we eschew `returns.context` in
# favour of `dependency_injector` for more readable code.
#
# References:
#
# * https://docs.python.org/3/library/asyncio-task.html
# * https://github.com/dry-python/returns#better-async-composition
#
# (soft) design goals:
#
# 1: We're explictly targeting PBSPro but want to avoid choices that make it
#    hard to switch to some other batch system. The minimal assumptions we
#    make are that:
#    a) the job id is returned on stdout when a job is submitted
#    b) some kind of sentinel file is created at the end of the job
#    c) job exit status can be queried after the job is done
# 2: Avoid data structures with rich behavior, BUT...
# 3: ... as far as possible keep all behavior that depends on details of which
#    batch system is used out of the module-level functions. So that'll end up
#    in the dataclasses.


class Container(containers.DeclarativeContainer):
    """Dependency injection.

    Loads default config from adjacent 'config.yml' file.
    """

    config = providers.Configuration(
        yaml_files=[path.join(path.dirname(__file__), 'config.yml')]
    )
    logging = providers.Resource(
        logging.basicConfig,
        stream=sys.stdout,
        level=config.log.level,
        format=config.log.format,
    )
    job_throttle = providers.Singleton(
        Throttler,
        rate_limit=config.submission.rate_limit,
        period=config.submission.period,
    )


@dataclass(kw_only=True)
class JobSpec:
    """Specifies a PBS batch job.

    Options not managed by explicit attributes may be supplied as-is via
    `extras` e.g. `extras='-l ngpus=1'`.

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
    error_path: Optional[str] = None
    extras: Optional[str] = None

    def __str__(self) -> str:
        """Representation as the `qsub` command for submitting the job.

        `cmd` is not included and is assumed to be passed via stdin.
        """
        return (
            'qsub'
            + f' -l mem="{self.mem}"'
            + f' -l ncpus="{self.ncpus}"'
            + f' -l walltime="{self.walltime}"'
            + f' -N "{self.name}"'
            + (f' -q "{self.queue}"' if self.queue else '')
            + (f' -e "{self.error_path}"' if self.error_path else '')
            + (f' {self.extras}' if self.extras else '')
        )


@dataclass(kw_only=True)
class Job:
    """Describes a submitted PBS job."""

    jobid: str
    name: str
    complete_on_file: str

    @classmethod
    def default_error_path(cls, jobid, jobname):
        """Default path to the error file for a submitted job.

        The format will depend on how the batch system is configured but
        usually includes the jobid and job name.
        """
        return path.join(getcwd(), f'{jobname}.e{jobid.split(".")[0]}')

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
        details = json.loads(_decode(output))['Jobs'][self.jobid]
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
@inject
async def submit(
    jobspec: JobSpec, throttle: Throttler = Provide[Container.job_throttle]
) -> Job:
    """Submit a job to the scheduler."""
    async with throttle:
        proc = await asyncio.create_subprocess_shell(
            str(jobspec), stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
        stdout, stderr = await proc.communicate(input=_encode(jobspec.cmd))
        if proc.returncode:
            raise RuntimeError(f'{jobspec}: {_decode(stderr)}')
        jobid = _decode(stdout)
        job = Job(
            jobid=jobid,
            name=jobspec.name,
            complete_on_file=jobspec.error_path
            or Job.default_error_path(jobid, jobspec.name),
        )
        LOGGER.info('Submitted %s', job)
        return job


@future_safe
@inject
async def wait_till_done(
    job: Job,
    interval: int = Provide[Container.config.polling.interval],
    timeout: int = Provide[Container.config.polling.timeout],
) -> Job:
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
                if proc.returncode:
                    raise RuntimeError(f'{job}: {_decode(stderr)}')
                success, msg = job.status_check(stdout)
                if not success:
                    raise RuntimeError(f'{job}: {msg}')
                return job
    except asyncio.TimeoutError as tee:
        raise RuntimeError(f'{job}: timed out after {timeout}s') from tee


async def handle(result: FutureResultE[Job]):
    """Do something with the pipeline result."""
    # Here we just log the result. In practice you'd probably want to e.g.
    # create a file, update a db record etc.
    match await result:
        case IOFailure(ex):
            LOGGER.error(ex)
        case IOSuccess(Success(job)):
            LOGGER.info('%s succeeded', job.jobid)


def _decode(byts: bytes, encoding='utf-8') -> str:
    """Decode bytes and strip whitespace"""
    return byts.decode(encoding).strip()


def _encode(text: str, encoding='utf-8') -> bytes:
    """Encode text"""
    return text.encode(encoding)
