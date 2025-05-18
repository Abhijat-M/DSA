class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count=0
        curr_sum=0
        prefix={0:1}

        for num in nums:
            curr_sum+=num
            required_sum= curr_sum -k
            count += prefix.get(required_sum, 0)
            prefix[curr_sum] = prefix.get(curr_sum, 0) + 1
            
        return count
        