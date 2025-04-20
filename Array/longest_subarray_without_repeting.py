class LongestDistinctSubarray:

    @staticmethod
    def longest_subarray(arr: list[int]) -> tuple[int, list[int]]:
        """Finds the length and the longest subarray with all distinct elements.

        Uses a sliding window approach to achieve O(n) time complexity.

        Args:
            arr: A list of integers.

        Returns:
            A tuple containing:
             - The length of the longest subarray with all distinct elements.
             - The longest subarray itself.
        """
        n = len(arr)
        if n == 0:
            return 0, []

        max_length = 0
        start = 0           
        result_start_index = 0 
        window_elements = set()

        for end in range(n):
            current_element = arr[end]

            while current_element in window_elements:
                window_elements.remove(arr[start])
                start += 1

            window_elements.add(current_element)

            current_length = end - start + 1
            if current_length > max_length:
                max_length = current_length
                result_start_index = start 

        longest_sub = arr[result_start_index : result_start_index + max_length]

        return max_length, longest_sub
    

    def test_longest_subarray(self):
        print("Running tests...")

        assert LongestDistinctSubarray.longest_subarray([1, 2, 3, 4, 5]) == (5, [1, 2, 3, 4, 5]), "Test case 1 failed"
        assert LongestDistinctSubarray.longest_subarray([1, 2, 2, 3, 4]) == (3, [2, 3, 4]), "Test case 2 failed"
        assert LongestDistinctSubarray.longest_subarray([1, 2, 3, 4, 5, 6]) == (6, [1, 2, 3, 4, 5, 6]), "Test case 3 failed"
        assert LongestDistinctSubarray.longest_subarray([1]) == (1, [1]), "Test case 4 failed"
        assert LongestDistinctSubarray.longest_subarray([]) == (0, []), "Test case 5 failed: Empty list"
        assert LongestDistinctSubarray.longest_subarray([5] * 10) == (1, [5]), "Test case 6 failed: All same elements"
        assert LongestDistinctSubarray.longest_subarray([1, 2, 3, 2, 1]) == (3, [1, 2, 3]), "Test case 7 failed"

        print("All test cases passed!")



if __name__ == "__main__":
    test_runner = LongestDistinctSubarray()
    test_runner.test_longest_subarray()

    print("\nDocstring of longest_subarray:")
    print(LongestDistinctSubarray.longest_subarray.__doc__)