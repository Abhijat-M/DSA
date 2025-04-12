class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return self.binarysearchrecursive(nums, target, 0, len(nums) - 1)

    def binarysearchrecursive(self, nums: List[int], target: int, left_index: int, right_index: int) -> int:
        if left_index > right_index:
            return left_index
        
        mid_index = left_index + (right_index - left_index) // 2
        mid_val = nums[mid_index]

        if mid_val == target:
            return mid_index
        elif mid_val < target:
            return self.binarysearchrecursive(nums, target, mid_index+1, right_index)
        elif mid_val > target:
            return self.binarysearchrecursive(nums, target, left_index, mid_index-1)