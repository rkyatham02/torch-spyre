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

"""
End-to-end tests for SpyreAllocator → FlexAllocator path through PyTorch's public tensor API.

These tests verify that torch.empty(size, device="spyre") correctly allocates device memory,
that tensors going out of scope trigger ReportAndDelete to free memory, and that sequential
allocate/free cycles leave the allocator in a consistent state.
"""

import gc
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
)


def get_allocator_stats():
    """Get current allocator statistics from SpyreAllocator."""
    # Ensure torch.spyre is initialized
    if not torch.spyre.is_initialized():
        torch.spyre._lazy_init()
    stats = torch.spyre._spyre_get_allocator_stats(0)
    return {
        "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
        "num_allocs": stats.get("allocation.all.current", 0),
    }


@instantiate_parametrized_tests
class TestAllocatorE2E(TestCase):
    """End-to-end tests for SpyreAllocator → FlexAllocator integration."""

    def setUp(self):
        """Reset allocator stats before each test."""
        super().setUp()
        # Force garbage collection to ensure clean state
        gc.collect()
        # Ensure torch.spyre is initialized
        if not torch.spyre.is_initialized():
            torch.spyre._lazy_init()
        torch.spyre._spyre_reset_accumulated_stats(0)
        torch.spyre._spyre_reset_peak_stats(0)

    def tearDown(self):
        """Clean up after each test."""
        gc.collect()
        super().tearDown()

    def test_basic_allocation(self):
        """
        Test 1: Basic allocation
        Verify that torch.empty((N,), device="spyre") returns a valid tensor
        with non-null storage and correct size.
        """
        N = 1024

        # Get initial stats
        initial_stats = get_allocator_stats()

        # Allocate tensor
        tensor = torch.empty((N,), device="spyre", dtype=torch.float32)

        # Verify tensor properties
        self.assertEqual(tensor.device.type, "spyre")
        self.assertEqual(tensor.shape, (N,))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertIsNotNone(tensor.data_ptr())
        self.assertGreater(tensor.data_ptr(), 0)

        # Verify storage is non-null
        self.assertIsNotNone(tensor.untyped_storage())
        self.assertGreater(tensor.untyped_storage().data_ptr(), 0)

        # Verify allocator stats increased
        current_stats = get_allocator_stats()
        self.assertGreater(
            current_stats["allocated_bytes"], initial_stats["allocated_bytes"]
        )
        self.assertGreater(current_stats["num_allocs"], initial_stats["num_allocs"])

        # Expected allocation size (N * sizeof(float32) = N * 4 bytes)
        expected_bytes = N * 4
        allocated_bytes = (
            current_stats["allocated_bytes"] - initial_stats["allocated_bytes"]
        )
        # Allow for alignment padding (FlexAllocator aligns to DEVICE_ALIGNMENT)
        self.assertGreaterEqual(allocated_bytes, expected_bytes)

    def test_automatic_deallocation(self):
        """
        Test 2: Automatic deallocation
        Allocate tensor in a scope, let it go out of scope, force GC,
        verify the block is freed (allocator free space increases).
        """
        N = 2048

        # Get initial stats
        initial_stats = get_allocator_stats()

        # Allocate tensor in a scope
        torch.empty((N,), device="spyre", dtype=torch.float32)

        # Verify allocation happened
        stats_during = get_allocator_stats()
        self.assertGreater(
            stats_during["allocated_bytes"], initial_stats["allocated_bytes"]
        )
        self.assertGreater(stats_during["num_allocs"], initial_stats["num_allocs"])

        # Force garbage collection to trigger ReportAndDelete
        gc.collect()

        # Verify deallocation happened
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["allocated_bytes"],
            initial_stats["allocated_bytes"],
            "Memory should be freed after tensor goes out of scope",
        )
        self.assertEqual(
            final_stats["num_allocs"],
            initial_stats["num_allocs"],
            "Allocation count should return to initial value",
        )

    # def test_explicit_deletion(self):
    #     """
    #     Test 2b: Explicit deletion with del
    #     Verify that explicitly deleting a tensor triggers deallocation.
    #     """
    #     N = 1024

    #     initial_stats = get_allocator_stats()

    #     # Allocate tensor
    #     tensor = torch.empty((N,), device="spyre", dtype=torch.float32)
    #     stats_after_alloc = get_allocator_stats()
    #     self.assertGreater(stats_after_alloc['allocated_bytes'], initial_stats['allocated_bytes'])

    #     # Explicitly delete tensor
    #     del tensor
    #     gc.collect()

    #     # Verify deallocation
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'],
    #                     "Memory should be freed after explicit deletion")

    # def test_sequential_alloc_free_cycles(self):
    #     """
    #     Test 3: Sequential alloc/free
    #     Allocate and free 100 tensors of the same size sequentially,
    #     verify allocator state is clean (no leaked blocks, free space matches initial).
    #     """
    #     N = 512
    #     num_cycles = 100

    #     initial_stats = get_allocator_stats()

    #     for i in range(num_cycles):
    #         # Allocate tensor
    #         tensor = torch.empty((N,), device="spyre", dtype=torch.float32)

    #         # Verify allocation
    #         self.assertIsNotNone(tensor.data_ptr())
    #         self.assertGreater(tensor.data_ptr(), 0)

    #         # Delete tensor
    #         del tensor
    #         gc.collect()

    #         # Verify deallocation after each cycle
    #         current_stats = get_allocator_stats()
    #         self.assertEqual(current_stats['allocated_bytes'], initial_stats['allocated_bytes'],
    #                        f"Memory leak detected at cycle {i+1}")
    #         self.assertEqual(current_stats['num_allocs'], initial_stats['num_allocs'],
    #                        f"Allocation count mismatch at cycle {i+1}")

    #     # Final verification
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'],
    #                     "Memory leaked after 100 sequential alloc/free cycles")
    #     self.assertEqual(final_stats['num_allocs'], initial_stats['num_allocs'],
    #                     "Allocation count leaked after 100 sequential alloc/free cycles")

    # def test_varying_sizes_random_order(self):
    #     """
    #     Test 4: Varying sizes
    #     Allocate tensors of different sizes (small, medium, large),
    #     free in random order, verify consistent state.
    #     """
    #     # Define test sizes: small, medium, large
    #     sizes = [
    #         128,      # small: 512 bytes
    #         4096,     # medium: 16 KB
    #         262144,   # large: 1 MB
    #         128,      # small again
    #         8192,     # medium-large: 32 KB
    #         524288,   # very large: 2 MB
    #     ]

    #     initial_stats = get_allocator_stats()

    #     # Allocate all tensors
    #     tensors = []
    #     for size in sizes:
    #         tensor = torch.empty((size,), device="spyre", dtype=torch.float32)
    #         self.assertIsNotNone(tensor.data_ptr())
    #         tensors.append(tensor)

    #     # Verify all allocations happened
    #     stats_after_alloc = get_allocator_stats()
    #     self.assertGreater(stats_after_alloc['allocated_bytes'], initial_stats['allocated_bytes'])
    #     self.assertEqual(stats_after_alloc['num_allocs'] - initial_stats['num_allocs'], len(sizes))

    #     # Shuffle tensors for random-order deallocation
    #     random.shuffle(tensors)

    #     # Free tensors in random order
    #     for tensor in tensors:
    #         del tensor
    #         gc.collect()

    #     # Clear the list
    #     tensors.clear()
    #     gc.collect()

    #     # Verify all memory is freed
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'],
    #                     "Memory leaked after random-order deallocation")
    #     self.assertEqual(final_stats['num_allocs'], initial_stats['num_allocs'],
    #                     "Allocation count mismatch after random-order deallocation")

    # def test_zero_size_allocation(self):
    #     """
    #     Test 5: Zero-size allocation
    #     Verify that torch.empty((0,), device="spyre") does not crash
    #     and behavior matches CPU allocator semantics.
    #     """
    #     initial_stats = get_allocator_stats()

    #     # Allocate zero-size tensor
    #     tensor = torch.empty((0,), device="spyre", dtype=torch.float32)

    #     # Verify tensor properties
    #     self.assertEqual(tensor.device.type, "spyre")
    #     self.assertEqual(tensor.shape, (0,))
    #     self.assertEqual(tensor.numel(), 0)

    #     # Zero-size allocations should return nullptr and not allocate memory
    #     # This matches CPU allocator behavior
    #     current_stats = get_allocator_stats()
    #     self.assertEqual(current_stats['allocated_bytes'], initial_stats['allocated_bytes'],
    #                     "Zero-size allocation should not allocate memory")

    #     # Delete tensor
    #     del tensor
    #     gc.collect()

    #     # Verify no memory leak
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'])

    # def test_multi_dimensional_tensors(self):
    #     """
    #     Test 6: Multi-dimensional tensors
    #     Verify allocation works correctly for multi-dimensional tensors.
    #     """
    #     shapes = [
    #         (10, 20),           # 2D
    #         (5, 10, 15),        # 3D
    #         (2, 3, 4, 5),       # 4D
    #     ]

    #     initial_stats = get_allocator_stats()

    #     for shape in shapes:
    #         tensor = torch.empty(shape, device="spyre", dtype=torch.float32)

    #         # Verify tensor properties
    #         self.assertEqual(tensor.device.type, "spyre")
    #         self.assertEqual(tensor.shape, shape)
    #         self.assertIsNotNone(tensor.data_ptr())

    #         # Calculate expected size
    #         numel = 1
    #         for dim in shape:
    #             numel *= dim
    #         expected_bytes = numel * 4  # float32 = 4 bytes

    #         # Verify allocation
    #         current_stats = get_allocator_stats()
    #         self.assertGreater(current_stats['allocated_bytes'], initial_stats['allocated_bytes'])

    #         # Clean up
    #         del tensor
    #         gc.collect()

    #         # Verify deallocation
    #         final_stats = get_allocator_stats()
    #         self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'])

    # def test_multiple_concurrent_allocations(self):
    #     """
    #     Test 7: Multiple concurrent allocations
    #     Allocate multiple tensors simultaneously and verify they all get unique storage.
    #     """
    #     N = 1024
    #     num_tensors = 10

    #     initial_stats = get_allocator_stats()

    #     # Allocate multiple tensors
    #     tensors = []
    #     data_ptrs = set()
    #     for i in range(num_tensors):
    #         tensor = torch.empty((N,), device="spyre", dtype=torch.float32)
    #         tensors.append(tensor)
    #         data_ptrs.add(tensor.data_ptr())

    #     # Verify all tensors have unique storage
    #     self.assertEqual(len(data_ptrs), num_tensors,
    #                     "All tensors should have unique storage addresses")

    #     # Verify allocator stats
    #     current_stats = get_allocator_stats()
    #     self.assertEqual(current_stats['num_allocs'] - initial_stats['num_allocs'], num_tensors)

    #     # Clean up
    #     tensors.clear()
    #     gc.collect()

    #     # Verify all memory freed
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'])

    # def test_large_allocation(self):
    #     """
    #     Test 8: Large allocation
    #     Verify that large allocations work correctly.
    #     """
    #     # Allocate 100 MB tensor
    #     N = 25 * 1024 * 1024  # 100 MB for float32

    #     initial_stats = get_allocator_stats()

    #     tensor = torch.empty((N,), device="spyre", dtype=torch.float32)

    #     # Verify allocation
    #     self.assertEqual(tensor.device.type, "spyre")
    #     self.assertEqual(tensor.numel(), N)
    #     self.assertIsNotNone(tensor.data_ptr())

    #     current_stats = get_allocator_stats()
    #     self.assertGreater(current_stats['allocated_bytes'], initial_stats['allocated_bytes'])

    #     # Clean up
    #     del tensor
    #     gc.collect()

    #     # Verify deallocation
    #     final_stats = get_allocator_stats()
    #     self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'])


# class TestAllocatorE2EMultithreaded(TestCase):
#     """Multi-threaded tests for SpyreAllocator → FlexAllocator integration."""

#     def setUp(self):
#         """Reset allocator stats before each test."""
#         super().setUp()
#         gc.collect()
#         torch.spyre._C._spyre_reset_accumulated_stats(0)
#         torch.spyre._C._spyre_reset_peak_stats(0)

#     def tearDown(self):
#         """Clean up after each test."""
#         gc.collect()
#         super().tearDown()

#     def test_concurrent_allocations_multithreaded(self):
#         """
#         Test 9: Multi-threaded concurrent allocations
#         Verify allocator is thread-safe with concurrent allocations from multiple threads.
#         """
#         N = 1024
#         num_threads = 4
#         allocations_per_thread = 25

#         initial_stats = get_allocator_stats()
#         errors = []

#         def allocate_and_free():
#             try:
#                 for _ in range(allocations_per_thread):
#                     tensor = torch.empty((N,), device="spyre", dtype=torch.float32)
#                     # Verify allocation
#                     assert tensor.data_ptr() > 0
#                     # Immediate cleanup
#                     del tensor
#                     gc.collect()
#             except Exception as e:
#                 errors.append(e)

#         # Create and start threads
#         threads = []
#         for _ in range(num_threads):
#             thread = threading.Thread(target=allocate_and_free)
#             threads.append(thread)
#             thread.start()

#         # Wait for all threads to complete
#         for thread in threads:
#             thread.join()

#         # Check for errors
#         self.assertEqual(len(errors), 0, f"Errors occurred in threads: {errors}")

#         # Force final cleanup
#         gc.collect()

#         # Verify no memory leaks
#         final_stats = get_allocator_stats()
#         self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'],
#                         "Memory leaked in multi-threaded test")

#     def test_sequential_cycles_multithreaded(self):
#         """
#         Test 10: Multi-threaded sequential cycles
#         Run sequential alloc/free cycles in multiple threads simultaneously.
#         """
#         N = 512
#         num_threads = 4
#         cycles_per_thread = 25

#         initial_stats = get_allocator_stats()
#         errors = []

#         def sequential_cycles():
#             try:
#                 for _ in range(cycles_per_thread):
#                     tensor = torch.empty((N,), device="spyre", dtype=torch.float32)
#                     assert tensor.data_ptr() > 0
#                     del tensor
#                     gc.collect()
#             except Exception as e:
#                 errors.append(e)

#         # Create and start threads
#         threads = []
#         for _ in range(num_threads):
#             thread = threading.Thread(target=sequential_cycles)
#             threads.append(thread)
#             thread.start()

#         # Wait for all threads
#         for thread in threads:
#             thread.join()

#         # Check for errors
#         self.assertEqual(len(errors), 0, f"Errors occurred in threads: {errors}")

#         # Force final cleanup
#         gc.collect()

#         # Verify no memory leaks
#         final_stats = get_allocator_stats()
#         self.assertEqual(final_stats['allocated_bytes'], initial_stats['allocated_bytes'],
#                         "Memory leaked in multi-threaded sequential cycles")


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
