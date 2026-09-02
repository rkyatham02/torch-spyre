# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile

import pytest
import torch
from torch_spyre import _C as _spyre_C
from torch.testing._internal.common_utils import TestCase

from test_prepare_kernel import TestPrepareKernel as tpk

# Stream ID of S_prep (kHostComputeStreamStartPerDevice = 65 in spyre_stream.cpp).
_HOST_COMPUTE_STREAM_START = 65


def _sync_all(device: torch.device) -> None:
    """Drain both S_dev and S_prep (ID 65). Safe to call with the split on or off."""
    torch.accelerator.synchronize(device)
    prep = _spyre_C.host_compute_stream_by_id(_HOST_COMPUTE_STREAM_START, device)
    prep.synchronize()


class TestSplitExecutionStructural(TestCase):
    """T1 and T2: run in both tracker-ON and tracker-OFF processes."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")

    def test_single_launch_split_does_not_crash(self):
        """T1 — A single ComputeOnHost launch completes without error.

        Fires one launch on a fresh stream, then drains both S_dev and S_prep.
        Passes if no exception is raised in either tracker-ON or tracker-OFF mode.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            inp = torch.zeros(64, 16, dtype=torch.float16, device=self.device)
            stream = torch.Stream(self.device)
            with stream:
                _spyre_C.launch_jobplan(job_plan, [inp])

            _sync_all(self.device)

    def test_single_launch_split_roles_and_no_event_steps(self):
        """T2 — Prepared plan is the bare triple: exactly 3 steps, correct types and roles.

        LOCK: no event or barrier steps must be injected into the plan.
          step 0: HostCompute  role=Prep
          step 1: H2D          role=Prep
          step 2: Compute      role=Dev
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            # Exactly 3 steps — no event/barrier steps were injected.
            self.assertEqual(job_plan.num_steps(), 3)

            self.assertEqual(
                [job_plan.get_step_type(i) for i in range(3)],
                ["HostCompute", "H2D", "Compute"],
            )
            self.assertEqual(
                [job_plan.get_step_stream_role(i) for i in range(3)],
                ["Prep", "Prep", "Dev"],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
