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
from typing import Any, Dict, Optional

from returns.future import future_safe
from returns.pipeline import flow
from returns.pointfree import bind


@dataclass
class JobSpec:
    """Specifies a PBS batch job."""
    cmd: str
    name: str
    ask_mem: str
    ask_ncpus: int
    ask_walltime: str
    path_err: Optional[str]=None
    path_out: Optional[str]=None


@dataclass
class Job:
    """Describes a submitted PBS job."""
    job_id: int 
    err_path: str


@future_safe
async def submit(jobspec: Optional[JobSpec|Dict[str, Any]]=None, **kwargs) -> Job:
    """Submit a job to the scheduler."""
    match jobspec:
        case JobSpec():
            pass
        case dict():
            jobspec = JobSpec(**jobspec)
        case _:
            jobspec = JobSpec(**kwargs)

    proc = await asyncio.create_subprocess_shell(f'''
        qsub \\
        -l mem={jobspec.ask_mem} \\
        -l ncpus={jobspec.ask_ncpus} \\
        -l walltime={jobspec.ask_walltime} \\
        -N "{jobspec.name}"''',
        stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await proc.communicate(input=jobspec.cmd.encode('utf-8'))
    if proc.returncode != 0:
        raise Exception(f'{jobspec}: {stderr.decode("utf-8")}')
    job_id = int(stdout.decode('utf-8').split('.')[0].strip())
    job_name = jobspec.name or 'STDIN'
    err_path = jobspec.path_err or f'{getcwd()}/{job_name}.e{job_id}'
    return Job(job_id, err_path)


@future_safe
async def wait_till_done(job: Job, waitsec=10) -> bool:
    """Wait until the job is finished."""
    while True:
        print(f'checking {job.job_id}...')
        if path.exists(job.err_path):
            print(f'{job.job_id} done!')
            return True
        await asyncio.sleep(waitsec)


async def main():

    jobs = [
        JobSpec(cmd='echo foo',
                name='a-sitting-on-a-gate',
                ask_mem='1MB',
                ask_ncpus=1,
                ask_walltime='00:01:00'),
        JobSpec(cmd='echo bar',
                name='haddocks-eyes',
                ask_mem='1MB',
                ask_ncpus=1,
                ask_walltime='00:01:00'),
    ]

    await asyncio.gather(*(flow(job, submit, bind(wait_till_done)) for job in jobs))
    

if __name__ == '__main__':
    asyncio.run(main())
