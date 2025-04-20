class TestTwoSum:

    @staticmethod 
    def two_sum(nums: list[int], target: int) -> list[int]:
        """Find two indices of numbers in the list that add up to the target."""
        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]
            num_to_index[num] = i
        return []

    def test_two_sum(self):
        assert TestTwoSum.two_sum([2, 7, 11, 15], 9) == [0, 1], "Test case 1 failed"
        assert TestTwoSum.two_sum([3, 2, 4], 6) == [1, 2], "Test case 2 failed"
        assert TestTwoSum.two_sum([3, 3], 6) == [0, 1], "Test case 3 failed"
        assert TestTwoSum.two_sum([1, 2, 3, 4, 5], 10) == [], "Test case 4 failed: No sum found"
        assert TestTwoSum.two_sum([], 10) == [], "Test case 5 failed: Empty list"
        assert TestTwoSum.two_sum([5], 10) == [], "Test case 6 failed: Single element"
        print("All test cases passed!")


if __name__ == "__main__":
    test_runner = TestTwoSum()
    test_runner.test_two_sum()