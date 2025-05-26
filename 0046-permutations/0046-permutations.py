class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def helper(unpro, pro=[]):
            if len(unpro) == 0:
                res.append(pro)
                return

            for i in range(len(pro) + 1):
                helper(unpro[1:], pro[:i] + [unpro[0]] + pro[i:])

            return

        helper(nums)
        return res
        