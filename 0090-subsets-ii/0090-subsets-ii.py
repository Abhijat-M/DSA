class Solution:
    
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:

        nums.sort()
        result = []
        
        def backtracking(nums, path):
            if not nums:
                result.append(path)
                return
    
            backtracking(nums[1:], path + [nums[0]])
            
            i = 1
            while i < len(nums) and nums[i] == nums[0]:
                i += 1
            backtracking(nums[i:], path)
        
        backtracking(nums, [])
        return result