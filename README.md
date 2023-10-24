# pbspy

Lightweight PBS job submission and monitoring

## Requirement(s)

* We want a [NiFi](https://nifi.apache.org/) custom processor to offload long-running and/or
computationally heavy tasks to our HPC.

## Implementation notes

By using an `asyncio` event loop we can submit and monitor hundreds of jobs
(or more) without blocking on IO or using multiple threads. By using `returns`
we can use FP style composition and not worry about checking for failures at
every point.

References:

* https://docs.python.org/3/library/asyncio-task.html
* https://github.com/dry-python/returns#better-async-composition
