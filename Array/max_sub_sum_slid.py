def max_subarray_sum(arr):
    if len(arr) < 3:
        raise ValueError("Array must have at least 3 elements")

    max_sum = sum(arr[:3])

    for i in range(1, len(arr) - 2):
        current_sum = sum(arr[i:i+3])
    
        if current_sum > max_sum:
            max_sum = current_sum

    return max_sum

def main():
    arr= eval(input("Enter array: "))
    print(max_subarray_sum(arr))

main()