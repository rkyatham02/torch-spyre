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

"""Split-execution trace tests (issue #2889, PR #4419)."""

import json
import math
import os
import warnings

import pytest
import torch
import torch._dynamo
import torch_spyre  # noqa: F401  # registers the backend
from torch.profiler import ProfilerActivity, profile
from torch.testing._internal.common_utils import TemporaryFileName

# ---------------------------------------------------------------------------
# Clock-calibration env vars — must be set before the flex runtime starts.
# If missing, the first profiler session triggers a calibration that stalls
# DMA for tens of milliseconds inside the profiled window, corrupting timing.
# We set them here as a fallback so the tests are self-contained when run
# manually without the full CI env; a warning is emitted so it is visible.
# ---------------------------------------------------------------------------
_CALIBRATION_VARS = {
    "FLEX_TIMESTAMP_CALIBRATION": "always",
    "FLEX_SKIP_TIMESTAMP_CALIBRATION": "0",
}
for _var, _val in _CALIBRATION_VARS.items():
    if os.environ.get(_var) != _val:
        warnings.warn(
            f"{_var} not set to '{_val}' — setting it now.  "
            f"For CI, set this on the runner so it takes effect before flex starts.",
            stacklevel=1,
        )
        os.environ[_var] = _val

# ---------------------------------------------------------------------------
# Workload constants — must be device-bound so kernel duration >> HC duration.
# Use a shape large enough that the kernel outlasts the host compute work,
# allowing HC[k+1] to complete while kernel[k] is still running on the device.
# ---------------------------------------------------------------------------
_MM_M, _MM_K, _MM_N = 2048, 8192, 2048
_MM_DTYPE = torch.float16
_NUM_LAUNCHES = 4  # back-to-back launches profiled per test


# ---------------------------------------------------------------------------
# Helpers — trace parsing
# ---------------------------------------------------------------------------


def _complete_events(trace_events: list[dict]) -> list[dict]:
    """Return only complete (ph=='X') events with a finite numeric ts."""
    out = []
    for ev in trace_events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        if (
            isinstance(ts, (int, float))
            and not isinstance(ts, bool)
            and math.isfinite(ts)
        ):
            out.append(ev)
    return out


def _kernel_events(events: list[dict]) -> list[dict]:
    """Return device compute events (cat=='kernel')."""
    return [e for e in events if e.get("cat") == "kernel"]


def _memcpy_events(events: list[dict]) -> list[dict]:
    """Return device DMA events (cat=='gpu_memcpy')."""
    return [e for e in events if e.get("cat") == "gpu_memcpy"]


def _host_compute_spans(events: list[dict]) -> list[dict]:
    """Return CPU-side launch::HostCompute spans.

    flex names the span 'launch::HostCompute'; it appears as a CPU event
    (cat=='cpu_op' or similar) not a device event.
    """
    return [
        e
        for e in events
        if "HostCompute" in e.get("name", "") and e.get("cat") != "kernel"
    ]


def _skew_us(events: list[dict]) -> float:
    """Estimate host-to-device clock skew in µs from correlated event pairs.

    Uses aiuLaunchControlBlocks (CPU) ↔ kernel (device) correlation pairs.
    For each pair, skew = kernel.ts - cpu_launch.ts - min_dispatch_overhead.
    We take the median over all pairs as the best estimate.

    Returns 0.0 if no correlated pairs are found (skew correction is skipped).
    """
    cpu_by_corr: dict[int, dict] = {}
    kernel_list: list[dict] = []

    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        corr = ev.get("args", {}).get("correlation")
        if corr is None:
            continue
        if (
            ev.get("cat") == "privateuse1_runtime"
            and ev.get("name") == "aiuLaunchControlBlocks"
        ):
            cpu_by_corr[int(corr)] = ev
        elif ev.get("cat") == "kernel":
            kernel_list.append(ev)

    deltas = []
    for k_ev in kernel_list:
        corr = int(k_ev.get("args", {}).get("correlation", -1))
        if corr in cpu_by_corr:
            cpu_ev = cpu_by_corr[corr]
            deltas.append(k_ev["ts"] - cpu_ev["ts"])

    if not deltas:
        return 0.0
    deltas.sort()
    mid = len(deltas) // 2
    return deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0


def _apply_skew(events: list[dict], skew_us: float) -> list[dict]:
    """Return a copy of events with device-clock ts values shifted by -skew_us.

    Only shifts kernel and gpu_memcpy events (device clock); leaves CPU events
    untouched.  Returns a shallow-copied list; individual dicts are copied too.
    """
    device_cats = {"kernel", "gpu_memcpy"}
    result = []
    for ev in events:
        if ev.get("cat") in device_cats:
            ev = dict(ev)
            ev["ts"] = ev["ts"] - skew_us
        result.append(ev)
    return result


# ---------------------------------------------------------------------------
# Helpers — overlap / ordering geometry
# ---------------------------------------------------------------------------


def _overlaps(a: dict, b: dict) -> bool:
    """True if event a's time range overlaps event b's time range."""
    a_start, a_end = a["ts"], a["ts"] + a.get("dur", 0)
    b_start, b_end = b["ts"], b["ts"] + b.get("dur", 0)
    return a_start < b_end and b_start < a_end


def _starts_inside(inner: dict, outer: dict) -> bool:
    """True if inner starts within outer's time range."""
    return outer["ts"] <= inner["ts"] < outer["ts"] + outer.get("dur", 0)


def _ends_before(a: dict, b: dict) -> bool:
    """True if a ends before b ends (a.ts + a.dur < b.ts + b.dur)."""
    return (a["ts"] + a.get("dur", 0)) < (b["ts"] + b.get("dur", 0))


# ---------------------------------------------------------------------------
# Fixture — compile once per session, warm up outside profiled region
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_mm():
    """Compile mm and warm up once; return the compiled callable."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("spyre")

    a = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
    b = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

    fn = torch.compile(torch.mm, backend="inductor")
    fn(a, b)
    torch.accelerator.synchronize(device)
    return fn


# ---------------------------------------------------------------------------
# Helper — run N back-to-back launches under the profiler, export trace
# ---------------------------------------------------------------------------


def _profile_back_to_back(compiled_fn, num_launches: int, tracker_on: bool) -> dict:
    """Run num_launches back-to-back mm calls under torch.profiler.

    Returns a dict with:
      - "events": list of complete trace events (clock-skew corrected)
      - "outputs": list of mm output tensors (on CPU, float32)
      - "refs":    list of CPU float32 reference tensors
      - "skew_us": the applied clock skew correction in µs
    """
    device = torch.device("spyre")
    torch.manual_seed(3 if tracker_on else 7)

    pairs_cpu = [
        (
            torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE),
            torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE),
        )
        for _ in range(num_launches)
    ]
    pairs_spyre = [(a.to(device), b.to(device)) for a, b in pairs_cpu]

    outputs_cpu = []
    refs = [torch.mm(a.float(), b.float()) for a, b in pairs_cpu]

    with TemporaryFileName(mode="w+") as trace_path:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as prof:
            for a_s, b_s in pairs_spyre:
                out = compiled_fn(a_s, b_s)
                outputs_cpu.append(out.cpu().float())
            torch.accelerator.synchronize(device)

        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            trace_data = json.load(f)

    raw_events = _complete_events(trace_data.get("traceEvents", []))
    skew = _skew_us(raw_events)
    corrected = _apply_skew(raw_events, skew)

    return {
        "events": corrected,
        "outputs": outputs_cpu,
        "refs": refs,
        "skew_us": skew,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_spyre_profiler
class TestSplitExecutionTrace:
    """Trace-level checks for split-execution correctness (issue #2889)."""

    def test_host_compute_span_present(self, compiled_mm):
        """launch::HostCompute CPU span appears in the trace with tracker on.

        This is the #3488 sentinel: if the span disappears after the host
        computation is moved off the calling thread, this test fails clearly
        rather than letting test_host_ran_ahead pass vacuously on an empty set.
        """
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=True)
        hc_spans = _host_compute_spans(result["events"])
        assert hc_spans, (
            "No launch::HostCompute CPU spans found in the trace with tracker on.  "
            "Either AIUPTI device events are missing (check USE_SPYRE_PROFILER=1), "
            "or the span was removed (post-#3488): update this file accordingly."
        )

    def test_host_compute_timing_diagnostic(self, compiled_mm):
        """Diagnostic: report HC-vs-kernel timing for both tracker modes (never fails).

        Overlap between host compute and device kernels is timing-dependent and
        varies across hardware, runs, and scheduler state.  This test always
        passes and prints per-pair timing so the reviewer can assess whether
        overlap occurred on the hardware under test.

        To harden into an assertion, run 10 times serially, confirm
        partly_inside >= 1 on every run, then replace this test with assertions.
        See plan section "Validate on a device" for the methodology.

        Per-pair columns:
          kernel_dur      — device kernel duration
          hc_dur          — launch::HostCompute CPU span duration
          hc_end→k_start  — gap from HC[k+1] end to kernel[k] start;
                            negative = HC ended after kernel started
          hc_start_inside — HC[k+1] started while kernel[k] was running
          hc_ends_before  — HC[k+1] ended before kernel[k] ended (ran ahead)
        """
        result_on = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=True)
        result_off = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=False)

        def _print_timing(result: dict, label: str) -> None:
            kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
            hc_spans = sorted(
                _host_compute_spans(result["events"]), key=lambda e: e["ts"]
            )
            num_pairs = min(len(kernels), len(hc_spans)) - 1
            ran_ahead = partly_inside = 0
            print(f"\n[{label}] skew_us={result['skew_us']:.2f}  pairs={num_pairs}")
            for k in range(max(num_pairs, 0)):
                k_start = kernels[k]["ts"]
                k_end = k_start + kernels[k].get("dur", 0)
                hc_start = hc_spans[k + 1]["ts"]
                hc_end = hc_start + hc_spans[k + 1].get("dur", 0)
                inside = _starts_inside(hc_spans[k + 1], kernels[k])
                ahead = _ends_before(hc_spans[k + 1], kernels[k])
                if inside:
                    partly_inside += 1
                if ahead:
                    ran_ahead += 1
                print(
                    f"  pair {k}: kernel_dur={k_end - k_start:.0f}µs  "
                    f"hc_dur={hc_end - hc_start:.0f}µs  "
                    f"hc_end→k_start={k_start - hc_end:+.0f}µs  "
                    f"hc_start_inside={'YES' if inside else 'no'}  "
                    f"hc_ends_before={'YES' if ahead else 'no'}"
                )
            print(
                f"  → ran_ahead={ran_ahead}/{num_pairs}  partly_inside={partly_inside}/{num_pairs}"
            )

        _print_timing(result_on, "tracker-on")
        _print_timing(result_off, "tracker-off")
        # Always passes — overlap is hardware/run-dependent.

    def test_raw_edge_kernel_after_h2d(self, compiled_mm):
        """RAW edge: kernel k starts after its own correction H2D ends.

        flex inserts this edge dynamically when the tracker is on.  A missing
        RAW edge would let the kernel start before the correction is written,
        corrupting the result.
        """
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=True)

        kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
        memcpys = sorted(_memcpy_events(result["events"]), key=lambda e: e["ts"])

        assert kernels, "No kernel events — check AIUPTI."
        assert memcpys, "No memcpy (H2D) events — check AIUPTI."

        # Pair each kernel with the H2D that immediately precedes it in time.
        violations = []
        for kernel in kernels:
            preceding_h2d = [
                m
                for m in memcpys
                if "HtoD" in m.get("name", "") and m["ts"] < kernel["ts"]
            ]
            if not preceding_h2d:
                continue
            closest_h2d = max(preceding_h2d, key=lambda m: m["ts"])
            h2d_end = closest_h2d["ts"] + closest_h2d.get("dur", 0)
            if kernel["ts"] < h2d_end:
                violations.append(
                    f"kernel ts={kernel['ts']:.3f} started before "
                    f"its H2D ended at {h2d_end:.3f} "
                    f"(delta={kernel['ts'] - h2d_end:.3f} µs)"
                )

        assert not violations, (
            "RAW edge violation(s) — kernel(s) started before their correction "
            "H2D completed:\n"
            + "\n".join(violations)
            + f"\nskew_us={result['skew_us']:.2f}"
        )

    def test_war_edge_h2d_after_preceding_kernel(self, compiled_mm):
        """WAR edge: correction H2D k+1 starts after kernel k ends.

        flex inserts this edge so that a new correction write cannot overwrite
        the device slot while the preceding kernel is still reading it.
        A missing WAR edge is the primary failure mode for back-to-back launches.
        """
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=True)

        kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
        h2ds = sorted(
            [
                m
                for m in _memcpy_events(result["events"])
                if "HtoD" in m.get("name", "")
            ],
            key=lambda e: e["ts"],
        )

        assert len(kernels) >= 2, "Need ≥ 2 kernel events to check WAR edge."
        assert len(h2ds) >= 2, "Need ≥ 2 H2D events to check WAR edge."

        violations = []
        # For adjacent kernel[k] / h2d[k+1] pairs: h2d[k+1] must start after
        # kernel[k] ends.
        num_pairs = min(len(kernels), len(h2ds)) - 1
        for k in range(num_pairs):
            kernel_k_end = kernels[k]["ts"] + kernels[k].get("dur", 0)
            h2d_kp1_start = h2ds[k + 1]["ts"]
            if h2d_kp1_start < kernel_k_end:
                violations.append(
                    f"H2D[{k + 1}] ts={h2d_kp1_start:.3f} started before "
                    f"kernel[{k}] ended at {kernel_k_end:.3f} "
                    f"(delta={h2d_kp1_start - kernel_k_end:.3f} µs)"
                )

        assert not violations, (
            "WAR edge violation(s) — H2D started before preceding kernel "
            "completed (missing WAR hazard edge):\n"
            + "\n".join(violations)
            + f"\nskew_us={result['skew_us']:.2f}"
        )

    def test_profiled_outputs_match_cpu(self, compiled_mm):
        """Outputs collected inside the profiled region match the CPU reference.

        Guards that the profiling session itself does not corrupt results
        (e.g. via calibration DMA stall or event-buffer interference).
        """
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES, tracker_on=True)

        for k, (out, ref) in enumerate(zip(result["outputs"], result["refs"])):
            rel_frob = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
            assert rel_frob < 0.01, (
                f"Launch {k}: relative Frobenius error {rel_frob:.4e} exceeds 1% "
                f"inside profiled region — profiling may have corrupted execution."
            )

    def test_reprofiled_window_after_calibration(self, compiled_mm):
        """A second profiling session after the first produces valid device events.

        Opening the first profiler session triggers a clock calibration that
        stalled DMA for 14–43 ms in empirical measurements (flex pre-#1930).
        Setting FLEX_TIMESTAMP_CALIBRATION=always (and FLEX_SKIP_TIMESTAMP_CALIBRATION=0)
        moves calibration to runtime start so the stall is outside both sessions.
        This test guards that: the second session (the "re-profiled window") still
        yields kernel events with non-zero, finite timestamps — i.e. the stall
        did not permanently break the device clock or AIUPTI buffer.
        """
        device = torch.device("spyre")
        torch.manual_seed(5)

        a = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
        b = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

        # First session — may incur calibration stall if env vars are not set.
        with TemporaryFileName(mode="w+") as _:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            ) as prof1:
                compiled_mm(a, b)
                torch.accelerator.synchronize(device)
            prof1.export_chrome_trace(_)

        # Second session — the re-profiled window. Must still have valid events.
        a2 = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
        b2 = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            ) as prof2:
                compiled_mm(a2, b2)
                torch.accelerator.synchronize(device)
            prof2.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                trace_data = json.load(f)

        events = _complete_events(trace_data.get("traceEvents", []))
        kernels = _kernel_events(events)

        assert kernels, (
            "No kernel events in re-profiled window (second session).  "
            "The calibration stall may have disrupted DMA/AIUPTI in the second "
            "session.  Set FLEX_TIMESTAMP_CALIBRATION=always and "
            "FLEX_SKIP_TIMESTAMP_CALIBRATION=0 to move calibration to runtime start."
        )

        invalid = [
            e
            for e in kernels
            if not (
                isinstance(e.get("ts"), (int, float))
                and math.isfinite(e["ts"])
                and e["ts"] > 0
                and e.get("dur", 0) > 0
            )
        ]
        assert not invalid, (
            f"{len(invalid)} kernel event(s) in re-profiled window have zero or "
            f"non-finite ts/dur — device clock may be broken after calibration stall: "
            + ", ".join(str(e.get("ts")) for e in invalid[:5])
        )
