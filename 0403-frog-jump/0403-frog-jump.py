class Solution:
    def canCross(self, stones: List[int]) -> bool:
        dp = [set() for _ in range(len(stones))]
        
        dp[0].add(1)
        
        for i in range(1,len(stones)):
            for j in range(i):
                diff = stones[i]-stones[j]
                if diff in dp[j]:
                    dp[i].add(diff-1)
                    dp[i].add(diff)
                    dp[i].add(diff+1)
                    
        return True if len(dp[-1])>0 else False
        