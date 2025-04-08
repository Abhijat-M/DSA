#You have been given an array, you need to find the maximum length of the subarray whose sum adds upto a total of 5

def max_length_subarray_with_sum(arr, target_sum):
    left = 0
    current_sum = 0
    max_length = 0

    for right in range(len(arr)):
        current_sum += arr[right]

        # Shrink the window from the left if the current sum exceeds the target
        while current_sum > target_sum and left <= right:
            current_sum -= arr[left]
            left += 1

        # Check if we have found a subarray with the exact target sum
        if current_sum == target_sum:
            max_length = max(max_length, right - left + 1)

    return max_length

arr = eval(input("ENTER ARRAY: "))
target_sum = 5
result = max_length_subarray_with_sum(arr, target_sum)
print("Maximum length of subarray with sum equal to 5:", result)