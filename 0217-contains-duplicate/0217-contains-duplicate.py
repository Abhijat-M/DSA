class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        from collections import Counter

        dic = Counter(nums)
        for i, j in zip(dic.keys(), dic.values()):
            if j>1:
                return True

        return False