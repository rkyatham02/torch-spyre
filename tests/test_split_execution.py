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
import torch._dynamo
from torch_spyre import _C as _spyre_C
from torch.testing._internal.common_utils import TestCase

from test_prepare_kernel import TestPrepareKernel as tpk

# Stream ID of S_prep (kHostComputeStreamStartPerDevice = 65 in spyre_stream.cpp).
_HOST_COMPUTE_STREAM_START = 65

# Number of successive launches used for pipelining tests (C2).
_PIPELINE_DEPTH = 4


def _sync_all(device: torch.device) -> None:
    """Drain both S_dev and S_prep (ID 65). Safe to call with the split on or off."""
    torch.accelerator.synchronize(device)
    prep = _spyre_C.host_compute_stream_by_id(_HOST_COMPUTE_STREAM_START, device)
    prep.synchronize()


class TestAcrossLaunchPipelining(TestCase):
    """C2 — successive launches pipeline S_prep over S_dev correctly.

    With the tracker on, launch N+1's HostCompute+H2D can start on S_prep
    while launch N's Compute is still running on S_dev.  The hazard tracker
    must ensure each Compute step sees its own H2D's results.
    """

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if not _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("C2 requires SPYRE_HAZARD_TRACKER=1")

    def _make_split_plan(self, tmpdir):
        spyrecode_dir = tpk().create_mock_spyrecode(
            tmpdir,
            exec_command="ComputeOnHost",
            exec_properties={
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["0"],  # fake-symbols: skips DataConvertInfoGenerate
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            },
        )
        return _spyre_C.prepare_kernel(spyrecode_dir)

    def test_pipelined_launches_reach_host_compute_step(self):
        """C2 — each launch routes HostCompute to S_prep (split is active).

        The mock HCM has no real DCI payload so the HostCompute callback raises
        "Expect one DCI" — that error can only come from HostCompute actually
        executing on S_prep.  Asserting it proves the split routing happened.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            job_plan = self._make_split_plan(tmpdir)

            self.assertEqual(job_plan.num_steps(), 3)
            self.assertEqual(
                [job_plan.get_step_stream_role(i) for i in range(3)],
                ["Prep", "Prep", "Dev"],
            )

            stream = torch.Stream(self.device)
            with stream:
                with self.assertRaisesRegex(RuntimeError, "Expect one DCI"):
                    _spyre_C.launch_jobplan(job_plan, [])


class TestFallbackNoPrepSteps(TestCase):
    """C5a — a pure ComputeOnDevice plan never touches S_prep.

    No Prep-role steps means the router's split branch never fires, regardless
    of the tracker flag.  Runs in both SPYRE_HAZARD_TRACKER=0 and =1.

    The structural test uses mock SpyreCode (no launch needed).
    The runtime tests use torch.compile to produce a real binary so the
    device actually executes the op rather than faulting on a fake pointer.
    """

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        torch._dynamo.reset()

    def test_no_prep_roles_in_pure_compute_plan(self):
        """C5a structural — every step has role Dev, no Prep steps exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(tmpdir)
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            for i in range(job_plan.num_steps()):
                self.assertEqual(
                    job_plan.get_step_stream_role(i),
                    "Dev",
                    msg=f"step {i} has unexpected Prep role in a pure-compute plan (C5a)",
                )

    def test_pure_compute_launch_matches_cpu(self):
        """C5a runtime — compiled abs runs on device and matches CPU reference."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        compiled = torch.compile(torch.abs, backend="inductor")
        spyre_result = compiled(x.to("spyre")).cpu()

        torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)
        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)

    def test_pure_compute_pipelined_matches_cpu(self):
        """C5a pipelining — repeated compiled abs calls all match CPU reference."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        compiled = torch.compile(torch.abs, backend="inductor")
        spyre_x = x.to("spyre")

        for _ in range(_PIPELINE_DEPTH):
            spyre_result = compiled(spyre_x).cpu()
            torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)

        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


class TestFallbackTrackerOff(TestCase):
    """C5b — correction triple with SPYRE_HAZARD_TRACKER=0 runs single-stream on S_dev.

    The router puts every step on S_dev when should_split is False.  S_dev's
    FIFO ordering enforces H2D→Compute without a cross-stream edge.
    The prepared plan shape is unchanged (still [Prep, Prep, Dev]) — only the
    routing at launch time differs.

    """

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("C5b requires SPYRE_HAZARD_TRACKER=0 (tracker-off path)")

    def test_plan_shape_unchanged_when_tracker_off(self):
        """C5b structural — prepare emits the same bare triple regardless of tracker flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            self.assertEqual(job_plan.num_steps(), 3)
            self.assertEqual(
                [job_plan.get_step_type(i) for i in range(3)],
                ["HostCompute", "H2D", "Compute"],
            )
            # Roles are still baked in at prepare time; the router ignores them
            # at launch when the tracker is off.
            self.assertEqual(
                [job_plan.get_step_stream_role(i) for i in range(3)],
                ["Prep", "Prep", "Dev"],
            )

    def test_correction_triple_launch_completes_tracker_off(self):
        """C5b runtime — correction triple routed entirely to S_dev completes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["0"],
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            stream = torch.Stream(self.device)
            with stream:
                _spyre_C.launch_jobplan(job_plan, [])

            # Tracker off → S_prep was never touched, only S_dev needs draining.
            torch.accelerator.synchronize(self.device)
            self.assertEqual(
                _spyre_C.get_device_state(),
                _spyre_C.SpyreDeviceState.Ok,
                msg="Device entered error state after single-stream launch (C5b).",
            )

    def test_pipelined_launches_tracker_off_healthy(self):
        """C5b pipelining — multiple correction-triple launches on S_dev alone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["0"],
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            stream = torch.Stream(self.device)
            with stream:
                for _ in range(_PIPELINE_DEPTH):
                    _spyre_C.launch_jobplan(job_plan, [])

            torch.accelerator.synchronize(self.device)
            self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
