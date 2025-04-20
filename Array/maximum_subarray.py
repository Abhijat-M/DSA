class TestMaxSubarraySum:
        
    @staticmethod
    def max_subarray_sum(nums: list[int]) -> int:
        '''Finds the maximum sum of a contiguous subarray by Kadane's Algorithm. 

            Args:
                nums: A list of integers.

            Returns:
                The maximum possible sum of a contiguous subarray.
                Returns 0 if the list is empty.
                Returns the largest number if all numbers are negative or zero.
        '''

        if not nums:
            return 0 

        max_so_far = nums[0]
        max_ending_here = nums[0]

        for i in range(1, len(nums)):
            num = nums[i]
            max_ending_here = max(num, max_ending_here + num)
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far



    def test_max_sub_sum(self):
        assert TestMaxSubarraySum.max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6, "Test case 1 failed"
        assert TestMaxSubarraySum.max_subarray_sum([1]) == 1, "Test case 2 failed"
        assert TestMaxSubarraySum.max_subarray_sum([5, 4, -1, 7, 8]) == 23, "Test case 3 failed"
        assert TestMaxSubarraySum.max_subarray_sum([-5, -1, -3]) == -1, "Test case 4 failed"
        assert TestMaxSubarraySum.max_subarray_sum([]) == 0, "Test case 5 failed: Empty list"
        assert TestMaxSubarraySum.max_subarray_sum([-1, -2, -3, -4]) == -1, "Test case 6 failed: All negative numbers"
        print("All test cases passed!")


if __name__ == "__main__":
    test_runner = TestMaxSubarraySum()
    test_runner.test_max_sub_sum()

    # Option 1: Print docstring of max_subarray_sum
    print("\nDocstring of max_subarray_sum:")
    print(TestMaxSubarraySum.max_subarray_sum.__doc__)
