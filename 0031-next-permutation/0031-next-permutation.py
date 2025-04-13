class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        
        if len(nums) == 1:
            return nums
        change = 0
        for i in range(len(nums)-2,-1,-1):
            if nums[i] < nums[i+1]:
                change = i
                break
        
        pter = len(nums)-1
        while pter > change:
            if nums[pter] > nums[change]:
                break
            pter-=1


        if pter != change:
            nums[change], nums[pter] = nums[pter], nums[change]
        else:
            nums[change], nums[-1] = nums[-1], nums[change]

        i = change+1
        j = len(nums)-1

        while i <= j:
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
                i+=1
            else:
                j-=1
        