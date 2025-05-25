class Solution: 
    def subsets(self, nums: list[int]) -> list[list[int]]:

        # result: list[list[int]] = [[]]

        # for num in nums:
        #     result += [curr + [num] for curr in result]

        # return result

        result: list = []

        def back(nums, curr):
            if len(nums) == 0:
                result.append(curr)
                return
            back(nums[1:], curr + [nums[0]])
            back(nums[1:], curr)

            return

        back(nums, [])
        return result