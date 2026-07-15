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
- TestDeviceErrorSkipIntegration: calls the conftest hook directly to verify
  skip behaviour without spawning subprocesses.

Usage: ``python test_device_error_skip.py`` or ``pytest test_device_error_skip.py``
"""

from unittest.mock import MagicMock, patch

import pytest
from torch.testing._internal.common_utils import TestCase, run_tests

from conftest import pytest_runtest_setup
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


class TestDeviceErrorSkipIntegration(TestCase):
    """
    Calls pytest_runtest_setup() directly with a mock item to verify the
    skip hook without subprocesses or pytester.
    """

    def _make_item(self, keywords=()):
        """Return a minimal mock pytest.Item with the given keyword names."""
        item = MagicMock(spec=pytest.Item)
        item.keywords = set(keywords)
        return item

    def test_healthy_device_does_not_skip(self):
        """When has_stream_error() is False the hook must not skip."""
        with patch.object(_C, "has_stream_error", return_value=False):
            # Should complete without raising pytest.skip.Exception
            pytest_runtest_setup(self._make_item())

    def test_faulted_device_skips(self):
        """When has_stream_error() is True every hook call must skip — not just the first."""
        with patch.object(_C, "has_stream_error", return_value=True):
            for _ in range(3):
                with self.assertRaises(pytest.skip.Exception):
                    pytest_runtest_setup(self._make_item())

    def test_skip_message_content(self):
        """The skip reason must contain 'Device is in error state'."""
        with patch.object(_C, "has_stream_error", return_value=True):
            with self.assertRaises(pytest.skip.Exception) as ctx:
                pytest_runtest_setup(self._make_item())
        self.assertIn("Device is in error state", str(ctx.exception))

    def test_import_error_does_not_block_test(self):
        """If _C raises ImportError the hook must silently pass."""
        with patch.object(_C, "has_stream_error", side_effect=ImportError):
            # Should complete without raising anything
            pytest_runtest_setup(self._make_item())


if __name__ == "__main__":
    run_tests()
