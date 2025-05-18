class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        sort_count_keys = sorted(count.keys(), key=lambda x: count[x])
        return sort_count_keys[-k:]