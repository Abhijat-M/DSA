import random

# QUICK SORT

def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# Test the quick_sort function

a=[random.randint(1,100) for i in range(10)]
print("UNSORTED: ",a)
print("SORTED: ",quick_sort(a))