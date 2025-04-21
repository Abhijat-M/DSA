from typing import List

class TimSort:
    _MIN_MERGE = 32

    def timSort(self, arr: List[int]) -> None:
        """
        Sorts the array in-place using a Timsort algorithm.
        """
        n = len(arr)
        if n < 2:
            return

        MIN_MERGE = TimSort._MIN_MERGE

        def calcMinRun(n):
            """Calculates the minimum run length for Timsort."""
            r = 0
            while n >= MIN_MERGE:
                r |= n & 1
                n >>= 1
            return n + r

        def insertionSort(arr, left, right):
            """Sorts array slice arr[left..right] using insertion sort."""
            for i in range(left + 1, right + 1):
                key = arr[i] # Store the element to be inserted
                j = i - 1
                # Move elements of arr[left..i-1] that are greater than key
                # one position ahead of their current position
                while j >= left and key < arr[j]:
                    arr[j + 1] = arr[j] # Shift element right
                    j -= 1
                arr[j + 1] = key # Insert the key in its correct position

        def merge(arr, l, m, r):
            """Merges two sorted subarrays arr[l..m] and arr[m+1..r]"""
            # Calculate lengths of the two subarrays
            len1, len2 = m - l + 1, r - m

            # Create temporary arrays using slicing (more Pythonic)
            left = arr[l : l + len1]
            right = arr[m + 1 : m + 1 + len2]

            # Pointers for left subarray, right subarray, and main array
            i, j, k = 0, 0, l

            # Merge the temp arrays back into arr[l..r]
            while i < len1 and j < len2:
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1

            # Copy remaining elements of left[], if any
            while i < len1:
                arr[k] = left[i]
                k += 1
                i += 1

            # Copy remaining elements of right[], if any
            while j < len2:
                arr[k] = right[j]
                k += 1
                j += 1
        

        minRun = calcMinRun(n)

        for start in range(0, n, minRun):
            end = min(start + minRun - 1, n - 1)
            insertionSort(arr, start, end)

        size = minRun
        while size < n:
        
            for left in range(0, n, 2 * size):
                # Find ending point of left sub array (mid)
                # mid + 1 is starting point of right sub array
                mid = min(n - 1, left + size - 1)

                # Find ending point of right sub array
                # Ensure 'right' doesn't exceed array bounds
                right = min((left + 2 * size - 1), (n - 1))

                # Merge sub arrays arr[left..mid] & arr[mid+1..right]
                # Only merge if mid < right
                if mid < right:
                    merge(arr, left, mid, right)

            size = 2 * size


    def test_sort_colors(self):
        print("Running tests...")

        arr1 = [2, 0, 2, 1, 1, 0]
        self.timSort(arr1)
        assert arr1 == [0, 0, 1, 1, 2, 2], "Test case 1 failed"

        arr2 = [0, 0, 1, 1, 2, 2]
        self.timSort(arr2)
        assert arr2 == [0, 0, 1, 1, 2, 2], "Test case 2 failed"

        arr3 = [2, 2, 1, 1, 0, 0]
        self.timSort(arr3)
        assert arr3 == [0, 0, 1, 1, 2, 2], "Test case 3 failed"

        arr4 = [1] * 10
        self.timSort(arr4)
        assert arr4 == [1] * 10, "Test case 4 failed"

        arr5 = []
        self.timSort(arr5)
        assert arr5 == [], "Test case 5 failed: Empty list"

        arr6 = [0] * 5 + [1] * 5 + [2] * 5
        self.timSort(arr6)
        assert arr6 == [0] * 5 + [1] * 5 + [2] * 5, "Test case 6 failed"

        print("All test cases passed!")


if __name__ == "__main__":
    tim_sort = TimSort()
    tim_sort.test_sort_colors()