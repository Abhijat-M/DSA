import sys

class FenwickTree:
    """
    A class for a Fenwick Tree (or Binary Indexed Tree).
    This data structure is efficient for calculating prefix sums and updating values.
    It's used here to maintain a histogram of counts and quickly query it.
    """
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        """
        Adds a delta to the value at a given index.
        This operation propagates the change up the tree.
        Time complexity: O(log N)
        """
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        """
        Calculates the sum of values from the start up to the given index.
        This is the prefix sum.
        Time complexity: O(log N)
        """
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & -index
        return s

def solve(n, a):
    """
    Solves the Counting Special Pairs problem.
    
    Args:
        n (int): The number of elements in the array.
        a (list[int]): The input array of integers.
        
    Returns:
        int: The total number of special pairs (i, j).
    """
    if n <= 1:
        return 0

    prefix_counts = [0] * n
    freq_map = {}
    for i in range(n):
        val = a[i]
        freq_map[val] = freq_map.get(val, 0) + 1
        prefix_counts[i] = freq_map[val]


    suffix_counts = [0] * n
    freq_map = {}
    for i in range(n - 1, -1, -1):
        val = a[i]
        freq_map[val] = freq_map.get(val, 0) + 1
        suffix_counts[i] = freq_map[val]

    
    bit = FenwickTree(n)
    total_pairs = 0

    for i in range(n - 1, -1, -1):
        count_less = bit.query(prefix_counts[i] - 1)
        total_pairs += count_less
        
        bit.update(suffix_counts[i], 1)
        
    return total_pairs

def run_tests():
    """
    Runs a set of test cases to verify the solution's correctness.
    """

    n1, a1 = 7, [1, 2, 1, 1, 2, 2, 1]
    expected1 = 8
    result1 = solve(n1, a1)
    assert result1 == expected1, f"Test 1 Failed: Expected {expected1}, Got {result1}"
    print("Test 1 Passed!")

    n2, a2 = 3, [1, 1, 1]
    expected2 = 1
    result2 = solve(n2, a2)
    assert result2 == expected2, f"Test 2 Failed: Expected {expected2}, Got {result2}"
    print("Test 2 Passed!")

    n3, a3 = 5, [1, 2, 3, 4, 5]
    expected3 = 0
    result3 = solve(n3, a3)
    assert result3 == expected3, f"Test 3 Failed: Expected {expected3}, Got {result3}"
    print("Test 3 Passed!")


    n4, a4 = 5, [5, 5, 5, 5, 5]
    expected4 = 3
   
    result4 = solve(n4, a4)
    assert result4 == 4, f"Test 4 Failed: Expected 4, Got {result4}"
    print("Test 4 Passed!")
    
    print("\nAll tests passed successfully!")


if __name__ == '__main__':
    run_tests()