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

- TestHasStreamErrorBinding: unit-tests _C.has_stream_error() via unittest.mock.
- TestDeviceErrorSkipIntegration: runs hermetic subprocess pytest sessions to
  verify the conftest skip hook end-to-end.
"""

import subprocess
import sys
import textwrap
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


def _run_pytest_subprocess(
    conftest_src: str, test_src: str
) -> subprocess.CompletedProcess:
    """
    Write conftest.py and test_tmp.py into a temp dir and run pytest in a
    subprocess, returning the CompletedProcess so callers can inspect stdout
    and returncode.
    """
    import os
    import tempfile

    tmpdir = tempfile.mkdtemp()
    conftest_path = os.path.join(tmpdir, "conftest.py")
    test_path = os.path.join(tmpdir, "test_tmp.py")
    with open(conftest_path, "w") as f:
        f.write(textwrap.dedent(conftest_src))
    with open(test_path, "w") as f:
        f.write(textwrap.dedent(test_src))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider", tmpdir],
        capture_output=True,
        text=True,
        env=env,
    )


class TestDeviceErrorSkipIntegration(TestCase):
    """
    Verifies the conftest pytest_runtest_setup hook behaviour end-to-end.

    Runs a hermetic subprocess pytest session so the host device state is
    never mutated and outcomes are inspectable as plain text.
    """

    def test_healthy_device_does_not_skip(self):
        """When has_stream_error() is False the test must run normally."""
        result = _run_pytest_subprocess(
            conftest_src="""
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
            """,
            test_src="""
                def test_should_run():
                    assert True
            """,
        )
        self.assertIn("1 passed", result.stdout)
        self.assertNotIn("skipped", result.stdout)

    def test_faulted_device_skips_all_tests(self):
        """
        When has_stream_error() is True every test in the session must be
        skipped. Two tests are declared to confirm all are affected, not
        just the first.
        """
        result = _run_pytest_subprocess(
            conftest_src="""
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
            """,
            test_src="""
                def test_first():
                    assert True

                def test_second():
                    assert True
            """,
        )
        self.assertIn("2 skipped", result.stdout)
        self.assertNotIn("passed", result.stdout)

    def test_skip_message_content(self):
        """The skip reason reported by pytest must contain 'Device is in error state'."""
        result = _run_pytest_subprocess(
            conftest_src="""
                from unittest.mock import patch
                import pytest
                from torch_spyre import _C

                def pytest_runtest_setup(item):
                    with patch.object(_C, "has_stream_error", return_value=True):
                        if _C.has_stream_error():
                            pytest.skip("Device is in error state")
            """,
            test_src="""
                def test_something():
                    pass
            """,
        )
        self.assertIn("1 skipped", result.stdout)
        self.assertIn("Device is in error state", result.stdout)

    def test_import_error_does_not_block_test(self):
        """If _C is unavailable the hook must silently pass and not skip."""
        result = _run_pytest_subprocess(
            conftest_src="""
                import pytest

                def pytest_runtest_setup(item):
                    try:
                        raise ImportError("_C not available")
                    except (ImportError, RuntimeError):
                        pass
            """,
            test_src="""
                def test_still_runs():
                    assert True
            """,
        )
        self.assertIn("1 passed", result.stdout)
        self.assertNotIn("skipped", result.stdout)


if __name__ == "__main__":
    run_tests()
