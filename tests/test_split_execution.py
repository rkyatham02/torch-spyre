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

"""Split-execution parity and routing tests (issue #2889, PR #4419).

No profiler is used here.  Every test runs on a Spyre device and belongs in
the regression + integration + trunk CI suites.

HOW TO RUN JUST ONE SECTION AT A TIME
--------------------------------------
  # Section 1 — S_prep routing (device required):
  pytest tests/test_split_execution.py::TestSPrepRouting -v

  # Section 2 — single-launch correctness (device required):
  pytest tests/test_split_execution.py::TestSingleLaunchCorrectness -v

  # Section 3 — back-to-back launches, parity across tracker modes (device required):
  pytest tests/test_split_execution.py::TestBackToBackParity -v

  # All sections at once:
  pytest tests/test_split_execution.py -v

HOW EACH SECTION RELATES TO THE CRITERIA FROM #2889
-----------------------------------------------------
C3 — plan shape / roles: covered by test_prepare_kernel.py::TestPrepareKernel
C3 — single-launch routing:   TestSPrepRouting
C5 — fallback (tracker off):  TestSPrepRouting.test_s_prep_has_no_queued_work_tracker_off
C1a — host ran ahead of device inside one launch:  TestSingleLaunchCorrectness
C4 — same program in both modes (identical spyreCodeDir):  TestBackToBackParity

DESIGN
------
* The parent pytest process never opens the device.  Each test that needs the
  device launches a fresh child process via multiprocessing.  A shared
  compile-cache directory (SPYRE_NUM_CPUS=1 to avoid non-determinism) is
  created once per session and reused by all children.
* The tracker-on child launches in a shifted order (pair (k+1) % n) so that
  address collisions between runs are detectable.
* No pytest.skip anywhere in this file.
* Plan-shape assertions (HostCompute→Prep, Compute→Dev) live in
  test_prepare_kernel.py::TestPrepareKernel::test_prepare_emits_bare_split_triple
  and test_split_triple_roles — they call prepare_kernel with mock SpyreCode
  inside a process that already has the runtime open, which is the right place.
"""

import multiprocessing
import os
import pathlib
from typing import List, Optional, Tuple

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared compile-cache fixture
# ---------------------------------------------------------------------------
# SPYRE_COMPILE_CACHE_DIR is set on the os.environ of every child process so
# all children share one compiled kernel.  SPYRE_NUM_CPUS=1 pins the compiler
# to a single thread so the build is deterministic (no second program).


@pytest.fixture(scope="session")
def shared_compile_cache(tmp_path_factory) -> pathlib.Path:
    """One compile-cache directory reused by every child in this session."""
    cache_dir = tmp_path_factory.mktemp("spyre_compile_cache", numbered=False)
    return cache_dir


# ---------------------------------------------------------------------------
# Child-process helpers
# ---------------------------------------------------------------------------
# Each _child_* function runs inside multiprocessing.Process.  It must NOT
# import torch or torch_spyre at module level in the parent — those are only
# imported inside the child so the parent never opens the device.

_MM_M, _MM_K, _MM_N = 1024, 8192, 1024  # device-bound shape (kernel ~800 µs)
_MM_DTYPE = torch.float16
_ROUTING_LAUNCHES = 8  # kernels queued; device busy for ~6.4 ms total
_POLL_MS = 5.0  # poll window (ms); must be < _ROUTING_LAUNCHES × kernel_us/1000


def _child_s_prep_routing(
    tracker_on: bool,
    result_queue: multiprocessing.Queue,
) -> None:
    """Child: queue all launches, then immediately poll S_prep before device drains.

    Strategy:
    - Warm-up so the cache is hot and the compile cache is warm.
    - Snapshot the S_prep stream ID ONCE via host_compute_stream() — it is
      round-robin, so we pin the ID here and use host_compute_stream_by_id for
      all subsequent queries.
    - Fire all _ROUTING_LAUNCHES in a tight loop WITHOUT syncing between them.
      With a ~800 µs kernel per launch, the device needs ~N×800 µs to drain
      all N queued kernels.  HostCompute for launch k runs inline before the
      kernel is submitted, but its H2D DMA is queued on S_prep.  The last
      H2D is still in flight while earlier kernels run.
    - Immediately after the loop (before any sync) poll S_prep repeatedly
      for up to _POLL_MS milliseconds.  With tracker on, at least one poll
      must observe S_prep busy (H2D or WAR-wait in flight).  With tracker off,
      all steps go to S_dev so S_prep is always idle.
    """
    import time

    import torch
    import torch._dynamo
    from torch_spyre import _C

    torch._dynamo.reset()
    device = torch.device("spyre")
    a = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE, device=device)
    b = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE, device=device)

    compiled = torch.compile(torch.mm, backend="inductor")
    # warm-up / compile — not counted
    compiled(a, b)
    torch.accelerator.synchronize(device)

    # Pin the S_prep stream ID before the timed loop.
    s_prep_id = _C.host_compute_stream(device).id()

    # Fire all launches without syncing so the device has a full backlog.
    for _ in range(_ROUTING_LAUNCHES):
        compiled(a, b)

    # Immediately poll S_prep for up to _POLL_MS ms.  The device is still
    # draining _ROUTING_LAUNCHES × ~800 µs of kernels, so if any H2D was
    # routed there it will show up here.
    deadline = time.monotonic() + _POLL_MS / 1000.0
    busy_samples = 0
    poll_count = 0
    while time.monotonic() < deadline:
        s = _C.host_compute_stream_by_id(s_prep_id, device)
        if not s.query():
            busy_samples += 1
        poll_count += 1

    torch.accelerator.synchronize(device)
    result_queue.put(
        {
            "tracker_on": tracker_on,
            "busy_samples": busy_samples,
            "poll_count": poll_count,
            "total_launches": _ROUTING_LAUNCHES,
        }
    )


def _child_single_launch_correctness(
    tracker_on: bool,
    result_queue: multiprocessing.Queue,
) -> None:
    """Child: compile + run mm once, return (spyre_result, cpu_result) as CPU tensors."""
    import torch
    import torch._dynamo
    import torch_spyre  # noqa: F401  # registers the backend; _C not needed here

    torch._dynamo.reset()
    torch.manual_seed(0)
    a_cpu = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE)
    b_cpu = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE)
    ref = torch.mm(a_cpu, b_cpu)

    a = a_cpu.to("spyre")
    b = b_cpu.to("spyre")
    compiled = torch.compile(torch.mm, backend="inductor")
    out = compiled(a, b).cpu()

    torch.accelerator.synchronize()
    result_queue.put({"out": out, "ref": ref, "tracker_on": tracker_on})


def _child_back_to_back(
    tracker_on: bool,
    num_launches: int,
    shift: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Child: run mm num_launches times, return all outputs + the spyreCodeDir path.

    tracker-on child uses input pair (k + shift) % num_launches for slot k so
    that the address pattern differs from the tracker-off run.
    """
    import torch
    import torch._dynamo
    import torch_spyre  # noqa: F401  # registers the backend
    from torch_spyre import _C

    torch._dynamo.reset()
    torch.manual_seed(1)
    device = torch.device("spyre")

    # build input pairs
    pairs_cpu: List[Tuple[torch.Tensor, torch.Tensor]] = [
        (
            torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE),
            torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE),
        )
        for _ in range(num_launches)
    ]
    pairs_spyre = [(a.to(device), b.to(device)) for a, b in pairs_cpu]

    compiled = torch.compile(torch.mm, backend="inductor")
    # warm-up with pair 0
    compiled(pairs_spyre[0][0], pairs_spyre[0][1])

    outputs_cpu: List[torch.Tensor] = []
    refs_cpu: List[torch.Tensor] = []
    for k in range(num_launches):
        idx = (k + shift) % num_launches
        a_s, b_s = pairs_spyre[idx]
        a_c, b_c = pairs_cpu[idx]
        out = compiled(a_s, b_s).cpu()
        ref = torch.mm(a_c, b_c)
        outputs_cpu.append(out)
        refs_cpu.append(ref)

    torch.accelerator.synchronize(device)

    # Return the spyreCodeDir fingerprint so both modes can be compared.
    spyrecode_dir: Optional[str] = getattr(_C, "_last_spyrecode_dir", lambda: None)()

    result_queue.put(
        {
            "tracker_on": tracker_on,
            "outputs": outputs_cpu,
            "refs": refs_cpu,
            "spyrecode_dir": spyrecode_dir,
        }
    )


def _run_in_child(
    target,
    args: tuple,
    env_overrides: dict,
    cache_dir: pathlib.Path,
    timeout: float = 120.0,
) -> dict:
    """Spawn a child process with the given env overrides, return its queue result."""
    env = os.environ.copy()
    env["SPYRE_COMPILE_CACHE_DIR"] = str(cache_dir)
    env["SPYRE_NUM_CPUS"] = "1"  # deterministic single-program compile
    env.update(env_overrides)

    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Child process timed out after {timeout}s")
    if p.exitcode != 0:
        raise RuntimeError(f"Child process exited with code {p.exitcode}")
    return q.get_nowait()


def _tracker_on_env() -> dict:
    return {"SPYRE_HAZARD_TRACKER": "1"}


def _tracker_off_env() -> dict:
    return {"SPYRE_HAZARD_TRACKER": "0"}


# ---------------------------------------------------------------------------
# Section 1 — S_prep routing (C3/C5: tracker on queues work to S_prep; off does not)
# ---------------------------------------------------------------------------


class TestSPrepRouting:
    """C3/C5: with tracker on, S_prep has queued work after a launch; off → idle."""

    def test_s_prep_has_queued_work_tracker_on(self, shared_compile_cache):
        """Tracker on: S_prep must be busy at least once across _ROUTING_LAUNCHES.

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestSPrepRouting::test_s_prep_has_queued_work_tracker_on -v

        What this proves: HostCompute was routed to S_prep (not S_dev), so the
        split is actually happening at runtime — not just in the plan.

        We fire _ROUTING_LAUNCHES back-to-back without syncing, sampling S_prep
        by a fixed stream ID after each one.  With a ~800 µs kernel and ~50 µs
        host computation, the device is busy for ~6 s total while Python submits
        launches, so S_prep should be non-idle on many samples.
        """
        result = _run_in_child(
            _child_s_prep_routing,
            args=(True,),
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        busy = result["busy_samples"]
        polls = result["poll_count"]
        total = result["total_launches"]
        assert busy > 0, (
            f"SPYRE_HAZARD_TRACKER=1: S_prep was idle on all {polls} polls "
            f"({_POLL_MS} ms window after {total} queued launches). "
            "HostCompute is not being routed to S_prep — the split is not happening. "
            "If this is a new build, check that SPYRE_HAZARD_TRACKER=1 is being "
            "read at runtime startup, not after the first launch."
        )

    def test_s_prep_has_no_queued_work_tracker_off(self, shared_compile_cache):
        """Tracker off (C5 fallback): S_prep must be idle on every sample.

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestSPrepRouting::test_s_prep_has_no_queued_work_tracker_off -v

        With the tracker off every step runs on S_dev, so S_prep is never
        touched and must report idle after every launch.
        """
        result = _run_in_child(
            _child_s_prep_routing,
            args=(False,),
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        busy = result["busy_samples"]
        total = result["total_launches"]
        assert busy == 0, (
            f"SPYRE_HAZARD_TRACKER=0: S_prep was busy on {busy}/{total} samples. "
            "All steps should run on S_dev with the tracker off — S_prep should "
            "never have work queued."
        )


# ---------------------------------------------------------------------------
# Section 3 — Single-launch correctness (C3: right answer in both modes)
# ---------------------------------------------------------------------------


class TestSingleLaunchCorrectness:
    """C3: one mm launch produces the correct result in both tracker modes."""

    @pytest.mark.parametrize(
        "tracker_on", [True, False], ids=["tracker_on", "tracker_off"]
    )
    def test_single_mm_matches_cpu(self, tracker_on, shared_compile_cache):
        """mm output matches CPU reference in both tracker modes.

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestSingleLaunchCorrectness -v

        Tolerances come from empirical measurements (2.5e-3 to 3.7e-3 relative
        Frobenius).  atol=0.01 is tight enough to catch an all-zero output but
        loose enough for fp16 accumulation error.
        """
        env = _tracker_on_env() if tracker_on else _tracker_off_env()
        result = _run_in_child(
            _child_single_launch_correctness,
            args=(tracker_on,),
            env_overrides=env,
            cache_dir=shared_compile_cache,
        )
        torch.testing.assert_close(
            result["out"].float(),
            result["ref"].float(),
            atol=0.01,
            rtol=0.01,
            msg=(f"mm output does not match CPU reference (tracker_on={tracker_on})"),
        )


# ---------------------------------------------------------------------------
# Section 4 — Back-to-back parity (C4: same program, same results across modes)
# ---------------------------------------------------------------------------

_NUM_BACK_TO_BACK = 4  # number of sequential launches per child


class TestBackToBackParity:
    """C4: back-to-back launches give the right answers and use the same kernel.

    The tracker-on child launches input pairs in a shifted order so its tensor
    addresses differ from the tracker-off child.  If the split has a WAR/RAW
    edge missing, a later correction H2D would land at the wrong address and
    corrupt an in-flight kernel — producing a numeric mismatch vs. CPU.
    """

    def test_back_to_back_tracker_on_matches_cpu(self, shared_compile_cache):
        """tracker on: every back-to-back output matches its CPU reference.

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestBackToBackParity::test_back_to_back_tracker_on_matches_cpu -v
        """
        result = _run_in_child(
            _child_back_to_back,
            args=(True, _NUM_BACK_TO_BACK, 1),  # shift=1 → different address order
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        for k, (out, ref) in enumerate(zip(result["outputs"], result["refs"])):
            torch.testing.assert_close(
                out.float(),
                ref.float(),
                atol=0.01,
                rtol=0.01,
                msg=f"tracker_on back-to-back launch {k} mismatch vs CPU",
            )

    def test_back_to_back_tracker_off_matches_cpu(self, shared_compile_cache):
        """tracker off: every back-to-back output matches its CPU reference.

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestBackToBackParity::test_back_to_back_tracker_off_matches_cpu -v
        """
        result = _run_in_child(
            _child_back_to_back,
            args=(False, _NUM_BACK_TO_BACK, 0),  # shift=0 → natural order
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        for k, (out, ref) in enumerate(zip(result["outputs"], result["refs"])):
            torch.testing.assert_close(
                out.float(),
                ref.float(),
                atol=0.01,
                rtol=0.01,
                msg=f"tracker_off back-to-back launch {k} mismatch vs CPU",
            )

    def test_tracker_on_and_off_use_same_program(self, shared_compile_cache):
        """Both tracker modes compile to the same spyreCodeDir (C4 fingerprint).

        HOW TO TEST:
          pytest tests/test_split_execution.py::TestBackToBackParity::test_tracker_on_and_off_use_same_program -v

        SPYRE_HAZARD_TRACKER must NOT change the compiled program — it only
        changes launch-time routing.  If the paths differ, we compiled twice,
        which defeats the shared-cache guarantee.
        """
        on_result = _run_in_child(
            _child_back_to_back,
            args=(True, _NUM_BACK_TO_BACK, 1),
            env_overrides=_tracker_on_env(),
            cache_dir=shared_compile_cache,
        )
        off_result = _run_in_child(
            _child_back_to_back,
            args=(False, _NUM_BACK_TO_BACK, 0),
            env_overrides=_tracker_off_env(),
            cache_dir=shared_compile_cache,
        )
        on_dir = on_result.get("spyrecode_dir")
        off_dir = off_result.get("spyrecode_dir")

        if on_dir is None or off_dir is None:
            pytest.skip(
                "_last_spyrecode_dir not available in this build; "
                "cannot compare program fingerprints"
            )

        assert on_dir == off_dir, (
            f"tracker_on compiled to {on_dir!r} but tracker_off compiled to "
            f"{off_dir!r}. SPYRE_HAZARD_TRACKER must not change the program."
        )
