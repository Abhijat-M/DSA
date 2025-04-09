def bucket_sort_1d(arr):
    if len(arr) == 0:
        return arr

    # Find the maximum and minimum values in the array
    min_val, max_val = min(arr), max(arr)
    bucket_count = len(arr)
    bucket_range = (max_val - min_val) / bucket_count

    # Create buckets
    buckets = [[] for _ in range(bucket_count)]

    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        if index == bucket_count:  # Handle edge case for max value
            index -= 1
        buckets[index].append(num)

    # Sort each bucket and concatenate
    sorted_array = []
    for bucket in buckets:
        sorted_array.extend(sorted(bucket))

    return sorted_array


def bucket_sort_2d(matrix):
    # Flatten the 2D array into 1D
    flattened = [num for row in matrix for num in row]

    # Sort the flattened array
    sorted_flattened = bucket_sort_1d(flattened)

    # Reconstruct the 2D array with the same dimensions
    rows, cols = len(matrix), len(matrix[0])
    sorted_matrix = []
    for i in range(rows):
        sorted_matrix.append(sorted_flattened[i * cols:(i + 1) * cols])

    return sorted_matrix


if __name__ == "__main__":
    # 1D array example
    array_1d = [0.42, 0.32, 0.23, 0.52, 0.25, 0.47]
    print("Sorted 1D array:", bucket_sort_1d(array_1d))

    # 2D array example
    array_2d = [
        [0.42, 0.32],
        [0.23, 0.52],
        [0.25, 0.47]
    ]
    print("Sorted 2D array:", bucket_sort_2d(array_2d))