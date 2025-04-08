import random
import time as t

# MERGE SORT ALGORITHM

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    return merge(merge_sort(left_half), merge_sort(right_half))

def merge(left, right):
    merg = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            merg.append(left[left_index])
            left_index += 1
        else:
            merg.append(right[right_index])
            right_index += 1

    merg += left[left_index:]
    merg += right[right_index:]

    return merg

# Test the merge_sort function
a=[random.randint(1,100) for i in range(10)]

print("UNSORTED: ",a)
t1=t.time()
print("SORTED: ",merge_sort(a))
t2=t.time()
print("TIME:",t2-t1)