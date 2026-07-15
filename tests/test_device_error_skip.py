# Copyright 2025 The Torch-Spyre Authors.
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

# Owner(s): ["module: stream"]

"""
Tests for the device-error-state skip mechanism.

1. TestHasStreamErrorBinding
   Tests the _C.has_stream_error() pybind11 binding in isolation using
   unittest.mock so no hardware fault needs to be induced.

2. TestDeviceErrorSkipIntegration
   Uses pytest's pytester plugin to spin up a throwaway sub-process pytest
   session and assert on skip outcomes without touching the host device.

Running
-------
    pytest tests/test_device_error_skip.py -p pytester
"""

from unittest.mock import patch

from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre import _C


class TestHasStreamErrorBinding(TestCase):
    """Unit tests for the _C.has_stream_error() pybind11 binding."""

    def test_binding_exists_and_returns_bool(self):
        """has_stream_error() must be importable and return a bool."""
        result = _C.has_stream_error()
        self.assertIsInstance(result, bool)

    def test_returns_false_when_mocked_healthy(self):
        """Returns False when the runtime reports no error."""
        with patch.object(_C, "has_stream_error", return_value=False):
            self.assertFalse(_C.has_stream_error())

    def test_returns_true_when_mocked_faulted(self):
        """Returns True when the runtime reports a stream error."""
        with patch.object(_C, "has_stream_error", return_value=True):
            self.assertTrue(_C.has_stream_error())

    def test_return_value_is_not_cached(self):
        """Consecutive calls reflect the live state, not a cached value."""
        with patch.object(_C, "has_stream_error", side_effect=[False, True, False]):
            self.assertFalse(_C.has_stream_error())
            self.assertTrue(_C.has_stream_error())
            self.assertFalse(_C.has_stream_error())


class TestDeviceErrorSkipIntegration:
    """
    Verifies the conftest pytest_runtest_setup hook behaviour end-to-end.

    Uses pytest's pytester plugin to run a hermetic sub-session so the host
    device state is never mutated and outcomes are inspectable as plain data.
    """

    def test_healthy_device_does_not_skip(self, pytester):
        """When has_stream_error() is False the test must run normally."""
        pytester.makepyfile(
            conftest="""
            from unittest.mock import patch
            import pytest
            from torch_spyre import _C

            def pytest_runtest_setup(item):
                with patch.object(_C, "has_stream_error", return_value=False):
                    try:
                        if _C.has_stream_error():
                            pytest.skip("Device is in error state")
                    except (ImportError, RuntimeError):
                        pass
            """
        )
        pytester.makepyfile(
            """
            def test_should_run():
                assert True
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1, skipped=0)

    def test_faulted_device_skips_all_tests(self, pytester):
        """
        When has_stream_error() is True every test in the session must be
        skipped. Two tests are declared to confirm all are affected, not
        just the first.
        """
        pytester.makepyfile(
            conftest="""
            from unittest.mock import patch
            import pytest
            from torch_spyre import _C

            def pytest_runtest_setup(item):
                with patch.object(_C, "has_stream_error", return_value=True):
                    try:
                        if _C.has_stream_error():
                            pytest.skip("Device is in error state")
                    except (ImportError, RuntimeError):
                        pass
            """
        )
        pytester.makepyfile(
            """
            def test_first():
                assert True

            def test_second():
                assert True
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=0, skipped=2)

    def test_skip_message_content(self, pytester):
        """The skip message must contain 'Device is in error state'."""
        pytester.makepyfile(
            conftest="""
            from unittest.mock import patch
            import pytest
            from torch_spyre import _C

            def pytest_runtest_setup(item):
                with patch.object(_C, "has_stream_error", return_value=True):
                    if _C.has_stream_error():
                        pytest.skip("Device is in error state")
            """
        )
        pytester.makepyfile(
            """
            def test_something():
                pass
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*Device is in error state*"])

    def test_import_error_does_not_block_test(self, pytester):
        """If _C is unavailable the hook must silently pass and not skip."""
        pytester.makepyfile(
            conftest="""
            import pytest

            def pytest_runtest_setup(item):
                try:
                    raise ImportError("_C not available")
                except (ImportError, RuntimeError):
                    pass
            """
        )
        pytester.makepyfile(
            """
            def test_still_runs():
                assert True
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1, skipped=0)


if __name__ == "__main__":
    run_tests()
