import timeit
import random
import numpy.core.multiarray
import math
import matplotlib.pyplot as plt # Optional: for plotting
import cProfile
import pstats
import io
import time # For sleep in profiling example if needed
from typing import List

# ==================================
# Your TimSort Implementation
# ==================================
class TimSort:
    _MIN_MERGE = 32 # Class attribute for the constant

    def timSort(self, arr: List[int]) -> None:
        """
        Sorts the array in-place using a Timsort algorithm.
        """
        n = len(arr)
        if n < 2:
            return # Already sorted if 0 or 1 element

        # Use the class attribute directly
        MIN_MERGE = TimSort._MIN_MERGE

        # --- Helper Functions Defined Inside (Consider moving to class level) ---
        # Note: Defining these here means they are recreated on every call.
        # It's often better practice to make them private methods (_calcMinRun, etc.)
        # or static methods if they don't need 'self'.

        def calcMinRun(n):
            """Calculates the minimum run length for Timsort."""
            r = 0
            # Check added to ensure n remains within reasonable bounds if MIN_MERGE is large relative to n
            while n >= MIN_MERGE:
                 # Check if n is non-negative before bitwise operations
                if n < 0: break
                r |= n & 1
                n >>= 1
            # Ensure returned value is at least 1 if n becomes 0
            return max(1, n + r)


        def insertionSort(arr, left, right):
            """Sorts array slice arr[left..right] using insertion sort."""
            for i in range(left + 1, right + 1):
                key = arr[i] # Store the element to be inserted
                j = i - 1
                # Move elements of arr[left..i-1] that are greater than key
                # one position ahead of their current position
                while j >= left and key < arr[j]:
                    arr[j + 1] = arr[j] # Shift element right
                    j -= 1
                arr[j + 1] = key # Insert the key in its correct position

        def merge(arr, l, m, r):
            """Merges two sorted subarrays arr[l..m] and arr[m+1..r]"""
            # Calculate lengths of the two subarrays
            len1, len2 = m - l + 1, r - m

            # Create temporary arrays using slicing (more Pythonic)
            # Handle potential edge cases where slicing might be empty
            left_temp = arr[l : l + len1]
            right_temp = arr[m + 1 : m + 1 + len2]


            # Pointers for left subarray, right subarray, and main array
            i, j, k = 0, 0, l

            # Merge the temp arrays back into arr[l..r]
            while i < len1 and j < len2:
                if left_temp[i] <= right_temp[j]:
                    arr[k] = left_temp[i]
                    i += 1
                else:
                    arr[k] = right_temp[j]
                    j += 1
                k += 1

            # Copy remaining elements of left_temp[], if any
            while i < len1:
                arr[k] = left_temp[i]
                k += 1
                i += 1

            # Copy remaining elements of right_temp[], if any
            while j < len2:
                arr[k] = right_temp[j]
                k += 1
                j += 1
        # --- End of Helper Functions ---

        minRun = calcMinRun(n)

        # Step 1: Sort individual subarrays of size minRun using Insertion Sort
        for start in range(0, n, minRun):
            # Ensure 'end' doesn't exceed array bounds
            end = min(start + minRun - 1, n - 1)
            # Check if start <= end before calling insertion sort
            if start <= end:
                insertionSort(arr, start, end)


        # Step 2: Start merging from size minRun (or current size). It will merge
        # to form size 2*minRun, then 4*minRun, 8*minRun and so on ....
        size = minRun
        while size < n:
            # Pick starting point of left sub array. We are going to merge
            # arr[left..left+size-1] and arr[left+size..left+2*size-1]
            # After every merge, we increase left by 2*size
            for left in range(0, n, 2 * size):
                # Find ending point of left sub array (mid)
                # mid+1 is starting point of right sub array
                # Ensure 'mid' doesn't exceed array bounds
                mid = min(n - 1, left + size - 1)


                # Find ending point of right sub array
                # Ensure 'right' doesn't exceed array bounds
                right = min((left + 2 * size - 1), (n - 1))

                # Merge sub arrays arr[left..mid] & arr[mid+1..right]
                # Only merge if the right subarray exists (mid < right)
                # and mid/right are valid indices
                if mid < right and mid >= left and right < n:
                     merge(arr, left, mid, right)


            # Increase size for next merge pass
            # Prevent potential infinite loop if size doesn't increase (e.g., if size starts at 0)
            if size == 0: # Should not happen if minRun calculation is correct
                 print("Warning: Merge size became 0. Breaking merge loop.")
                 break
            new_size = 2 * size
            if new_size <= size: # Check for overflow or non-increase
                 print(f"Warning: Merge size did not increase significantly ({size} -> {new_size}). Breaking.")
                 break
            size = new_size



    def run_tests(self):
        """Runs correctness tests for the timSort method."""
        print("Running correctness tests...")
        test_cases = {
            "Test Case 1 (Mixed)": ([2, 0, 2, 1, 1, 0], [0, 0, 1, 1, 2, 2]),
            "Test Case 2 (Sorted)": ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]),
            "Test Case 3 (Reversed)": ([2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2]),
            "Test Case 4 (All Same)": ([1] * 10, [1] * 10),
            "Test Case 5 (Empty)": ([], []),
            "Test Case 6 (Partially Sorted)": ([0]*5 + [1]*5 + [2]*5, [0]*5 + [1]*5 + [2]*5),
            "Test Case 7 (Single Element)": ([5], [5]),
            "Test Case 8 (Two Elements Sorted)": ([3, 7], [3, 7]),
            "Test Case 9 (Two Elements Unsorted)": ([7, 3], [3, 7]),
        }

        # Add a test with random data
        random_arr = [random.randint(0, 1000) for _ in range(100)]
        expected_random = sorted(random_arr)
        test_cases["Test Case 10 (Random)"] = (random_arr, expected_random)

        all_passed = True
        for name, (input_arr, expected_arr) in test_cases.items():
            # Create a copy to sort in-place
            arr_copy = input_arr[:]
            try:
                self.timSort(arr_copy)
                assert arr_copy == expected_arr, f"{name} failed: Expected {expected_arr}, got {arr_copy}"
                # print(f"{name}: Passed")
            except Exception as e:
                print(f"{name} failed with exception: {e}")
                print(f"  Input was: {input_arr}")
                all_passed = False

        if all_passed:
            print("All test cases passed!")
        else:
            print("Some test cases failed.")
        return all_passed

# ==================================
# Evaluation Code
# ==================================

def generate_random_list(size):
    """Generates a list of random integers."""
    # Generate numbers in a range related to the size to avoid excessive duplicates
    # for very large sizes, while still allowing some duplicates.
    max_val = max(size, 100) # Ensure a minimum range
    return [random.randint(0, max_val) for _ in range(size)]

def measure_execution_time(sorter_instance, input_generator, sizes, number=10, repeat=3):
    """
    Measures the execution time of a sorter's timSort method for various input sizes.

    Args:
        sorter_instance (TimSort): An instance of the TimSort class.
        input_generator (callable): A function that takes a size (int) and
                                    returns valid input data (a list).
        sizes (list[int]): A list of different input sizes to test.
        number (int): How many times to execute the sort in each trial.
                      Since the sort is in-place, we time copy+sort.
        repeat (int): How many times to repeat the timing trials.

    Returns:
        tuple: A tuple containing two lists: (measured_sizes, measured_times)
    """
    measured_times = []
    measured_sizes = []
    func_name = sorter_instance.timSort.__name__ # Get method name

    print(f"\nStarting time measurements for {func_name}...")
    print(f"Sizes: {sizes}")
    print(f"Timeit settings: number={number}, repeat={repeat}")
    print("-" * 40)

    for size in sizes:
        try:
            # Prepare setup code for timeit
            # Imports the class, creates an instance, generates original data
            setup_code = f"""
import random
from __main__ import TimSort, generate_random_list
sorter_instance = TimSort()
input_data = generate_random_list({size})
            """
            # Prepare statement: Copy the list and sort the copy (for in-place)
            stmt_code = "input_copy = input_data[:]; sorter_instance.timSort(input_copy)"

            # Perform the timing
            timer = timeit.Timer(stmt=stmt_code, setup=setup_code)
            times = timer.repeat(repeat=repeat, number=number)

            # Get minimum total time and calculate average time per execution
            min_time_total = min(times)
            avg_time_per_execution = min_time_total / number

            measured_sizes.append(size)
            measured_times.append(avg_time_per_execution)
            print(f"Size: {size:<10} | Avg Time/Exec: {avg_time_per_execution:.6f} seconds")

        except Exception as e:
            print(f"Error measuring size {size}: {e}")
            continue # Skip to next size if error occurs

    print("-" * 40)
    print("Time measurements complete.")
    return measured_sizes, measured_times



if __name__ == "__main__":
    sorter = TimSort()

    # --- 1. Correctness Tests ---
    sorter.run_tests()

    # --- 2. Time Complexity Analysis ---
    # Define input sizes for timing analysis
    # Use a range appropriate for O(n log n) - can go higher than O(n^2)
    # input_sizes = [10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 15000]
    input_sizes = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000] # Adjusted range


    # Adjust 'number' and 'repeat' based on expected speed.
    # For potentially faster sorts, increase 'number'. Let's start lower.
    sizes, times = measure_execution_time(
        sorter, generate_random_list, input_sizes, number=5, repeat=3
    )

    # --- Plotting Results (Requires matplotlib) ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, 'o-', label='Measured Timsort Time')

        # Optional: Plot theoretical curves for comparison (scaled)
        # Scale factors are rough estimates based on the last measured point
        if times and sizes:
            # O(n log n) scaling
            last_n = sizes[-1]
            if last_n > 1:
                scale_nlogn = times[-1] / (last_n * math.log(last_n))
                nlogn_theory = [scale_nlogn * s * math.log(s) if s > 1 else 0 for s in sizes]
                plt.plot(sizes, nlogn_theory, '--', label='Theoretical O(n log n) scale', alpha=0.7)

            # O(n) scaling (for comparison, TimSort best case)
            scale_n = times[-1] / last_n
            n_theory = [scale_n * s for s in sizes]
            plt.plot(sizes, n_theory, '--', label='Theoretical O(n) scale', alpha=0.7)

        plt.xlabel("Input Size (n)")
        plt.ylabel("Average Execution Time (seconds)")
        plt.title("Empirical Time Complexity of TimSort Implementation")
        plt.legend()
        plt.grid(True)
        # Consider log scales if times vary greatly or to see polynomial degrees
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()

    except ImportError:
        print("\nMatplotlib not installed. Cannot plot timing results.")
        print("Install using: pip install matplotlib")
    except Exception as plot_err:
        print(f"\nError during plotting: {plot_err}")


    # --- 3. Profiling Analysis ---
    print("\n" + "=" * 60)
    print("Profiling a single run of timSort (size=1000)")
    print("=" * 60)

    # Prepare data for profiling run
    profile_data_size = 1000
    profile_data = generate_random_list(profile_data_size)
    profile_data_copy = profile_data[:] # Copy as it sorts in-place

    # Create profiler instance
    profiler = cProfile.Profile()

    # Enable, run the sort, disable
    print(f"--- Running timSort on list of size {profile_data_size} under profiler ---")
    profiler.enable()
    sorter.timSort(profile_data_copy) # Sort the copy
    profiler.disable()
    print("--- Function execution finished ---")

    # Print stats sorted by cumulative time
    print("\n--- Profiling Statistics (sorted by cumulative time) ---")
    profiler.print_stats(sort='cumulative')

    # Optional: Save stats to file
    # profiler.dump_stats('timsort_profile.prof')
    # print("\nProfile statistics saved to 'timsort_profile.prof'")

    # Check if the profiled sort actually worked (optional sanity check)
    # print(f"Profiled list sorted correctly: {profile_data_copy == sorted(profile_data)}")

