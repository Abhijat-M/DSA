import sys

def solve_compression(n, a):
    if not a:
        return []
        
    unique_sorted_a = sorted(list(set(a)))
    
    rank_map = {val: i for i, val in enumerate(unique_sorted_a)}
    
    b = [0] * n
    for i in range(n):
        b[i] = rank_map[a[i]]
        
    return b

def run_tests():
    n1, a1 = 5, [3, 3, 1, 6, 1]
    expected1 = [1, 1, 0, 2, 0]
    result1 = solve_compression(n1, a1)
    assert result1 == expected1, f"Test 1 Failed: Expected {expected1}, Got {result1}"

    n2, a2 = 4, [10, 20, 30, 40]
    expected2 = [0, 1, 2, 3]
    result2 = solve_compression(n2, a2)
    assert result2 == expected2, f"Test 2 Failed: Expected {expected2}, Got {result2}"

    n3, a3 = 5, [100, 100, 100, 100, 100]
    expected3 = [0, 0, 0, 0, 0]
    result3 = solve_compression(n3, a3)
    assert result3 == expected3, f"Test 3 Failed: Expected {expected3}, Got {result3}"

    n4, a4 = 4, [40, 30, 20, 10]
    expected4 = [3, 2, 1, 0]
    result4 = solve_compression(n4, a4)
    assert result4 == expected4, f"Test 4 Failed: Expected {expected4}, Got {result4}"
    
    n5, a5 = 0, []
    expected5 = []
    result5 = solve_compression(n5, a5)
    assert result5 == expected5, f"Test 5 Failed: Expected {expected5}, Got {result5}"
    
    n6, a6 = 6, [10, -5, 10, 0, -5, 20]
    expected6 = [2, 0, 2, 1, 0, 3]
    result6 = solve_compression(n6, a6)
    assert result6 == expected6, f"Test 6 Failed: Expected {expected6}, Got {result6}"
    
    print("All test cases passed successfully!")

if __name__ == '__main__':
    run_tests()

    # try:
    #     n_input_str = sys.stdin.readline()
    #     if n_input_str and n_input_str.strip():
    #         n_input = int(n_input_str)
    #         a_input = [int(sys.stdin.readline()) for _ in range(n_input)]
    #         result_b = solve_compression(n_input, a_input)
    #         for val in result_b:
    #             print(val)
    # except (IOError, ValueError):
    #     pass
