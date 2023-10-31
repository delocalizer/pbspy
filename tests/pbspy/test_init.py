"""
Unit tests for pbspy package __init__
"""
import unittest
from unittest.mock import patch

from returns.io import IOFailure, IOSuccess
from returns.result import Failure, Success
import pbspy


class TestJobSpec(unittest.TestCase):
    """Tests for JobSpec dataclass."""

    def test_str(self):
        """
        Test that `str(instance)` generates job submission command of the
        expected form.
        """
        jobspec = pbspy.JobSpec(
            cmd='echo hello',
            mem='1MB',
            ncpus=1,
            walltime='00:01:00',
            name='foo',
            queue='testing',
            error_path='/path/to/err.file',
            extras='-l ngpus=1',
        )
        self.assertEqual(
            str(jobspec),
            (
                'qsub -l mem="1MB" -l ncpus="1" -l walltime="00:01:00" '
                '-N "foo" -q "testing" -e "/path/to/err.file" -l ngpus=1'
            ),
        )


class TestJob(unittest.TestCase):
    """Tests for Job dataclass."""

    def test_default_error_path(self):
        """Test that default error file path is of expected format."""
        with patch('pbspy.getcwd', return_value='/home/user1'):
            self.assertEqual(
                pbspy.Job.default_error_path('1234.pbsserver', 'my-jerb'),
                '/home/user1/my-jerb.e1234',
            )

    def test_str(self):
        """Test that `str(instance)` returns human-readable representation."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='my-jerb',
            complete_on_file='/path/to/err.file',
        )
        self.assertEqual(str(job), 'my-jerb [1234.pbsserver]')

    def test_status_cmd(self):
        """Test that status command is of the expected format."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='my-jerb',
            complete_on_file='/path/to/err.file',
        )
        self.assertEqual(job.status_cmd(), 'qstat -f -x -F json 1234.pbsserver')

    def test_status_check_fail(self):
        """Test that parsing of successful status check works as expected."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        # heavily pruned output from qstat -f -x -F json
        stdout = b'''{
            "Jobs":{
                "1234.pbsserver":{
                    "Job_Name":"haddocks-eyes",
                    "comment":"Job run at Wed Oct 25 at 09:19 and failed",
                    "Exit_status":127
                }
            }
        }'''
        self.assertEqual(
            job.status_check(stdout),
            (
                False,
                ('Job run at Wed Oct 25 at 09:19 and failed'),
            ),
        )

    def test_status_check_success(self):
        """Test that parsing of successful status check works as expected."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        stdout = b'''{
            "Jobs":{
                "1234.pbsserver":{
                    "Job_Name":"haddocks-eyes",
                    "comment":"Job run at Wed Oct 25 at 09:18 and finished",
                    "Exit_status":0
                }
            }
        }'''
        self.assertEqual(
            job.status_check(stdout),
            (
                True,
                ('Job run at Wed Oct 25 at 09:18 and finished'),
            ),
        )


class TestAsyncModuleFunctions(unittest.IsolatedAsyncioTestCase):
    """Tests for the async module functions."""

    container = pbspy.Container()
    # short poll for faster tests
    container.config.polling.interval.from_value(1)
    container.wire(modules=['pbspy', __name__])

    async def test_submit_failure(self):
        """Test that a failed job submission fails in the expected way."""
        with patch('pbspy.JobSpec.__str__', return_value='not-qsub'):
            jobspec = pbspy.JobSpec(
                cmd='echo hello',
                mem='1MB',
                ncpus=1,
                walltime='00:01:00',
                name='foo',
            )
            result = await pbspy.submit(jobspec)
            self.assertIsInstance(result, IOFailure)
            self.assertIn('not-qsub', str(result))

    async def test_submit_success(self):
        """Test that a successful job submission succeeds in the expected way."""
        jobspec = pbspy.JobSpec(
            cmd='echo hello',
            mem='1MB',
            ncpus=1,
            walltime='00:01:00',
            name='foo',
        )
        with patch('pbspy.JobSpec.__str__', return_value=''), patch(
            'asyncio.subprocess.Process.communicate',
            return_value=(b'1234.pbsserver', b''),
        ):
            result = await pbspy.submit(jobspec)
            self.assertIsInstance(result, IOSuccess)
            self.assertIn('foo [1234.pbsserver]', str(result))

    async def test_wait_till_done_succeeded(self):
        """Test that successful job succeeds in the expected way."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        stdout = b'''{
            "Jobs":{
                "1234.pbsserver":{
                    "Job_Name":"haddocks-eyes",
                    "comment":"Job run at Wed Oct 25 at 09:18 and finished",
                    "Exit_status":0
                }
            }
        }'''
        with patch('pbspy.path.exists', return_value=True), patch(
            'asyncio.subprocess.Process.communicate', return_value=(stdout, b'')
        ), patch('asyncio.subprocess.Process.returncode', 0):
            result = await pbspy.wait_till_done(job)
            self.assertIsInstance(result, IOSuccess)
            self.assertIn('haddocks-eyes [1234.pbsserver]', str(result))

    async def test_wait_till_done_failed(self):
        """Test that failed job fails in the expected way."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        stdout = b'''{
            "Jobs":{
                "1234.pbsserver":{
                    "Job_Name":"haddocks-eyes",
                    "comment":"Job run at Wed Oct 25 at 09:19 and failed",
                    "Exit_status":127
                }
            }
        }'''
        with patch('pbspy.path.exists', return_value=True), patch(
            'asyncio.subprocess.Process.communicate', return_value=(stdout, b'')
        ), patch('asyncio.subprocess.Process.returncode', 0):
            result = await pbspy.wait_till_done(job)
            self.assertIsInstance(result, IOFailure)
            self.assertIn(
                'Job run at Wed Oct 25 at 09:19 and failed',
                str(result),
            )

    async def test_wait_till_done_timedout(self):
        """Test that timed-out job fails in the expected way."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        with self.container.config.polling.timeout.override(1):
            result = await pbspy.wait_till_done(job)
            self.assertIsInstance(result, IOFailure)
            self.assertIn(
                'haddocks-eyes [1234.pbsserver]: timed out after 1s',
                str(result),
            )
