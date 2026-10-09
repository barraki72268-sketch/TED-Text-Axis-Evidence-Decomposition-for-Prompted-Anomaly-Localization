import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.axis_run import require_live_gpu_allocation, run


class AxisRunTests(unittest.TestCase):
    def test_refuse_environment_without_allocated_visible_gpu_before_reading_workspace(self):
        for env in [{}, {'SLURM_JOB_ID':'123', 'CUDA_VISIBLE_DEVICES':''},
                    {'SLURM_JOB_ID':'123', 'CUDA_VISIBLE_DEVICES':'-1'}]:
            with patch.dict(os.environ, env, clear=True), patch('reproduction.axis_run.subprocess.run') as launch:
                with self.assertRaisesRegex(RuntimeError, 'actual GPU Slurm'):
                    run(Path('/missing'), Path('/missing'))
                launch.assert_not_called()

    def test_live_job_must_have_gpu_and_current_host(self):
        env = {'SLURM_JOB_ID':'123', 'CUDA_VISIBLE_DEVICES':'0'}
        job = 'JobId=123 JobState=RUNNING AllocTRES=cpu=8,gres/gpu=1 NodeList=gpu-host Reservation=test'
        with patch.dict(os.environ, env, clear=True), patch('reproduction.axis_run.socket.gethostname', return_value='gpu-host'):
            with patch('reproduction.axis_run.subprocess.run', side_effect=[SimpleNamespace(stdout=job), SimpleNamespace(stdout='gpu-host\n')]):
                self.assertEqual(require_live_gpu_allocation()['job_id'], '123')
            for invalid in [job.replace('RUNNING','COMPLETED'), job.replace('gres/gpu=1','gres/gpu=0')]:
                with patch('reproduction.axis_run.subprocess.run', return_value=SimpleNamespace(stdout=invalid)):
                    with self.assertRaisesRegex(RuntimeError, 'running GPU'):
                        require_live_gpu_allocation()
            with patch('reproduction.axis_run.subprocess.run', side_effect=[SimpleNamespace(stdout=job), SimpleNamespace(stdout='other-host\n')]):
                with self.assertRaisesRegex(RuntimeError, 'not allocated'):
                    require_live_gpu_allocation()
