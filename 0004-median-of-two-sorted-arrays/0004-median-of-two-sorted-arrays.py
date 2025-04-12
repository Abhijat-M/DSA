class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        
        numbers = sorted(nums1 + nums2)
        n = len(numbers)
    
        if n % 2 == 1:
            median = numbers[n // 2]
        else:
            mid_index1 = n // 2 - 1
            mid_index2 = n // 2
            median = (numbers[mid_index1] + numbers[mid_index2]) / 2

        return float(median)