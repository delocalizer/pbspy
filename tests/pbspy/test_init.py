"""
Unit tests for pbspy package __init__
"""
import unittest
from unittest.mock import patch

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

    def test_status_check(self):
        """Test that parsing of status command output works as expected."""
        job = pbspy.Job(
            jobid='1234.pbsserver',
            name='haddocks-eyes',
            complete_on_file='/path/to/err.file',
        )
        # heavily pruned output from qstat -f -x -F json
        output = b'''{
            "Jobs":{
                "1234.pbsserver":{
                    "Job_Name":"haddocks-eyes",
                    "Job_Owner":"conradL@hpcapp01.adqimr.ad.lan",
                    "Resource_List":{
                        "mem":"1mb",
                        "ncpus":1,
                        "nodect":1,
                        "place":"pack",
                        "select":"1:mem=1mb:ncpus=1",
                        "walltime":"00:01:00"
                    },
                    "comment":"Job run at Wed Oct 25 at 09:18 on (hpcnode070:mem=1024kb:ncpus=1) and finished",
                    "Exit_status":0
                }
            }
        }'''
        self.assertEqual(
            job.status_check(output),
            (
                True,
                (
                    'Job run at Wed Oct 25 at 09:18 '
                    'on (hpcnode070:mem=1024kb:ncpus=1) and finished'
                ),
            ),
        )

class TestModuleFunctions(unittest.TestCase):
    """Tests for the module functions."""
