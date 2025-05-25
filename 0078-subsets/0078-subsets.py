class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        out=[]
        out.append([])
        if len(nums)==1:
            out.append(nums)
            return out
        for i in range(1,len(nums)):
            for j in combinations(nums,i):
                out.append(list(j))
        out.append(nums)
        return out