def getSumAbsoluteDifferences(nums: list[int]) -> list[int]:
    ans=list()
    for i in range(0, len(nums)):
        sum=0
        for j in range(0, len(nums)):
            sum= sum+ abs(nums[i]-nums[j])
        
        ans.append(sum)

    return ans

a=eval(input("Enter the list of numbers: "))
print(getSumAbsoluteDifferences(a))