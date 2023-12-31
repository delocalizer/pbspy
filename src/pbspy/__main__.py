"""Lightweight PBS job submission and monitoring."""
import argparse
import json
import importlib.metadata

from pbspy import Container, JobSpec, run


def example_job_spec() -> str:
    """Show the JobSpec dataclass schema."""
    return (
        '{\n'
        + '\n'.join(
            f'  "{v.name}": {v.type}' for k, v in JobSpec.__dataclass_fields__.items()
        )
        + '\n}'
    )


def job_spec_arg(j: str) -> JobSpec:
    """Load a JobSpec from serialized JSON."""
    try:
        return JobSpec(**json.loads(j))
    except Exception as err:
        print(err)
        print('Required JobSpec schema:\n', example_job_spec())
        raise argparse.ArgumentTypeError(f'invalid JobSpec: "{j}"')


def main():
    """Run as a CLI script.

    Positional arguments are JSON serialized JobSpec instances."""
    # a little demo app
    container = Container()
    container.init_resources()
    container.wire(modules=['pbspy', __name__])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'jobspec',
        nargs='+',
        type=job_spec_arg,
        metavar='JobSpec',
        help='JSON serialized JobSpec',
    )
    parser.add_argument(
        '--poll-interval', type=int, help='seconds to wait between job polling events'
    )
    parser.add_argument(
        '--poll-timeout',
        type=int,
        help='hours to wait for job to finish before giving up',
    )
    parser.add_argument(
        '--version', action='version', version=importlib.metadata.version('pbspy')
    )
    args = parser.parse_args()
    # set any custom config from CLI args, e.g. timeout
    if args.poll_interval:
        container.config.polling.interval.from_value(args.poll_interval)
    if args.poll_timeout:
        container.config.polling.timeout.from_value(3600 * args.poll_timeout)
    run(args.jobspec)


if __name__ == '__main__':
    main()
