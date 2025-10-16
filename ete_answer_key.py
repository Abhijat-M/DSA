# AAPS ETE Practice Questions - Python 3.8+ Solutions
# This file contains the Python code for all the questions in the provided list,
# ordered by topic as per the answer key.

import collections
import heapq
from typing import List, Optional

# =================================================================
# I. Arrays
# =================================================================

print("I. Arrays")
print("-" * 20)

# 1. Prefix Sum Array
class PrefixSum:
    """
    Class to handle prefix sum calculations for a given array.
    """
    def __init__(self, arr: List[int]):
        self.prefix_sum = [0] * len(arr)
        self.prefix_sum[0] = arr[0]
        for i in range(1, len(arr)):
            self.prefix_sum[i] = self.prefix_sum[i-1] + arr[i]

    def query(self, L: int, R: int) -> int:
        """
        Calculates the sum of elements in the range [L, R].
        """
        if L == 0:
            return self.prefix_sum[R]
        return self.prefix_sum[R] - self.prefix_sum[L-1]

# Example for Prefix Sum
arr1 = [1, 2, 3, 4, 5]
ps = PrefixSum(arr1)
print(f"1. Prefix Sum of {arr1} in range [1, 3]: {ps.query(1, 3)}") # Expected: 2+3+4 = 9

# 2. Equilibrium Index
def equilibrium_index(arr: List[int]) -> int:
    """
    Finds an index where the sum of left elements equals the sum of right elements.
    """
    total_sum = sum(arr)
    left_sum = 0
    for i, num in enumerate(arr):
        right_sum = total_sum - left_sum - num
        if left_sum == right_sum:
            return i
        left_sum += num
    return -1 # Return -1 if no equilibrium index is found

# Example for Equilibrium Index
arr2 = [-7, 1, 5, 2, -4, 3, 0]
print(f"2. Equilibrium Index of {arr2}: {equilibrium_index(arr2)}") # Expected: 3

# 3. Split Array into Equal Sum Prefix and Suffix
def can_split_array(arr: List[int]) -> bool:
    """
    Checks if an array can be split into two parts with equal sums.
    """
    total_sum = sum(arr)
    prefix_sum = 0
    for i in range(len(arr) - 1):
        prefix_sum += arr[i]
        if prefix_sum == total_sum - prefix_sum:
            return True
    return False

# Example for Split Array
arr3 = [1, 2, 3, 3, 2, 1]
print(f"3. Can {arr3} be split into equal sum parts? {can_split_array(arr3)}") # Expected: True

# 4. Two Sum (Sorted Array)
def two_sum_sorted(arr: List[int], target: int) -> List[int]:
    """
    Finds two numbers in a sorted array that add up to a target.
    """
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [] # Return empty list if no pair is found

# Example for Two Sum (Sorted)
arr4 = [2, 7, 11, 15]
target4 = 9
print(f"4. Two Sum in sorted {arr4} for target {target4}: {two_sum_sorted(arr4, target4)}") # Expected: [0, 1]

# 5. Two Sum (Unsorted Array)
def two_sum_unsorted(arr: List[int], target: int) -> List[int]:
    """
    Finds two numbers in an unsorted array that add up to a target using a hash map.
    """
    num_map = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []

# Example for Two Sum (Unsorted)
arr5 = [11, 2, 15, 7]
target5 = 9
print(f"5. Two Sum in unsorted {arr5} for target {target5}: {two_sum_unsorted(arr5, target5)}") # Expected: [1, 3]

# 6. Majority Element
def majority_element(arr: List[int]) -> int:
    """
    Finds the element that appears more than n/2 times using Boyer-Moore Voting Algorithm.
    """
    candidate, count = None, 0
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate

# Example for Majority Element
arr6 = [2, 2, 1, 1, 1, 2, 2]
print(f"6. Majority element in {arr6}: {majority_element(arr6)}") # Expected: 2

# 7. Next Permutation
def next_permutation(arr: List[int]) -> None:
    """
    Rearranges numbers into the lexicographically next greater permutation in-place.
    """
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] >= arr[i+1]:
        i -= 1
    
    if i >= 0:
        j = n - 1
        while arr[j] <= arr[i]:
            j -= 1
        arr[i], arr[j] = arr[j], arr[i]
    
    # Reverse the part of the array after index i
    left, right = i + 1, n - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# Example for Next Permutation
arr7 = [1, 2, 3]
next_permutation(arr7)
print(f"7. Next permutation of [1, 2, 3]: {arr7}") # Expected: [1, 3, 2]

# 8. Trapping Rainwater
def trap_rainwater(height: List[int]) -> int:
    """
    Calculates how much water can be trapped between bars.
    """
    if not height:
        return 0
    n = len(height)
    left_max = [0] * n
    right_max = [0] * n
    
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
        
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
        
    trapped_water = 0
    for i in range(n):
        trapped_water += min(left_max[i], right_max[i]) - height[i]
        
    return trapped_water

# Example for Trapping Rainwater
arr8 = [0,1,0,2,1,0,1,3,2,1,2,1]
print(f"8. Trapped rainwater in {arr8}: {trap_rainwater(arr8)}") # Expected: 6

# 9. Subarray Sum Equals K
def subarray_sum_equals_k(arr: List[int], k: int) -> int:
    """
    Finds the number of contiguous subarrays whose sum equals k.
    """
    count = 0
    current_sum = 0
    prefix_sum_map = {0: 1} # {sum: frequency}
    
    for num in arr:
        current_sum += num
        if current_sum - k in prefix_sum_map:
            count += prefix_sum_map[current_sum - k]
        prefix_sum_map[current_sum] = prefix_sum_map.get(current_sum, 0) + 1
        
    return count

# Example for Subarray Sum Equals K
arr9 = [1, 1, 1]
k9 = 2
print(f"9. Number of subarrays in {arr9} with sum {k9}: {subarray_sum_equals_k(arr9, k9)}") # Expected: 2

# 10. Median of Two Sorted Arrays
def median_of_two_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Finds the median of two sorted arrays.
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    x, y = len(nums1), len(nums2)
    low, high = 0, x
    
    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (x + y + 1) // 2 - partitionX
        
        maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
        minX = float('inf') if partitionX == x else nums1[partitionX]
        
        maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
        minY = float('inf') if partitionY == y else nums2[partitionY]
        
        if maxX <= minY and maxY <= minX:
            if (x + y) % 2 == 0:
                return (max(maxX, maxY) + min(minX, minY)) / 2
            else:
                return max(maxX, maxY)
        elif maxX > minY:
            high = partitionX - 1
        else:
            low = partitionX + 1
    # Should not happen if inputs are sorted arrays
    raise ValueError("Input arrays are not sorted.")

# Example for Median of Two Sorted Arrays
arr10_1 = [1, 3]
arr10_2 = [2]
print(f"10. Median of {arr10_1} and {arr10_2}: {median_of_two_sorted_arrays(arr10_1, arr10_2)}") # Expected: 2.0

# 11. Find the Element with Maximum Frequency
def max_frequency_element(arr: List[int]) -> int:
    """
    Finds the element that appears most frequently in an array.
    """
    if not arr:
        return None
    counts = collections.Counter(arr)
    return counts.most_common(1)[0][0]

# Example for Max Frequency Element
arr11 = [1, 2, 2, 3, 3, 3, 4]
print(f"11. Max frequency element in {arr11}: {max_frequency_element(arr11)}") # Expected: 3

# =================================================================
# II. Strings
# =================================================================

print("\nII. Strings")
print("-" * 20)

# 1. Longest Palindromic Substring
def longest_palindromic_substring(s: str) -> str:
    """
    Finds the longest palindromic substring in a given string.
    """
    if len(s) < 2 or s == s[::-1]:
        return s
    
    n = len(s)
    start, max_len = 0, 1
    
    # Expand from center
    for i in range(n):
        # Odd length palindromes
        l, r = i, i
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > max_len:
                start = l
                max_len = r - l + 1
            l -= 1
            r += 1
            
        # Even length palindromes
        l, r = i, i + 1
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > max_len:
                start = l
                max_len = r - l + 1
            l -= 1
            r += 1
            
    return s[start:start + max_len]

# Example for Longest Palindromic Substring
s1 = "babad"
print(f"1. Longest palindromic substring of '{s1}': '{longest_palindromic_substring(s1)}'") # Expected: "bab" or "aba"

# 2. Longest Common Prefix
def longest_common_prefix(strs: List[str]) -> str:
    """
    Finds the longest common prefix among a list of strings.
    """
    if not strs:
        return ""
    
    prefix = strs[0]
    for i in range(1, len(strs)):
        while strs[i].find(prefix) != 0:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

# Example for Longest Common Prefix
strs2 = ["flower","flow","flight"]
print(f"2. Longest common prefix of {strs2}: '{longest_common_prefix(strs2)}'") # Expected: "fl"

# 3. Edit Distance
def edit_distance(word1: str, word2: str) -> int:
    """
    Finds the minimum number of operations to convert word1 to word2.
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                   dp[i][j-1],      # Insertion
                                   dp[i-1][j-1])    # Substitution
    return dp[m][n]

# Example for Edit Distance
word1_3, word2_3 = "horse", "ros"
print(f"3. Edit distance between '{word1_3}' and '{word2_3}': {edit_distance(word1_3, word2_3)}") # Expected: 3

# =================================================================
# III. Linked Lists
# =================================================================

print("\nIII. Linked Lists")
print("-" * 20)

class ListNode:
    """
    Definition for a singly-linked list node.
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def print_list(node: ListNode):
    """Helper to print a linked list."""
    res = []
    while node:
        res.append(str(node.val))
        node = node.next
    print(" -> ".join(res))

# 1. Merge Two Sorted Linked Lists
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merges two sorted linked lists into one sorted list.
    """
    dummy = ListNode()
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
        
    current.next = l1 or l2
    return dummy.next

# Example for Merge Two Sorted Lists
l1_1 = ListNode(1, ListNode(2, ListNode(4)))
l2_1 = ListNode(1, ListNode(3, ListNode(4)))
merged_list = merge_two_lists(l1_1, l2_1)
print("1. Merged list: ", end="")
print_list(merged_list)

# 2. Remove N-th Node From End of List
def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    Removes the n-th node from the end of a list.
    """
    dummy = ListNode(0, head)
    fast = slow = dummy
    
    for _ in range(n + 1):
        fast = fast.next
        
    while fast:
        fast = fast.next
        slow = slow.next
        
    slow.next = slow.next.next
    return dummy.next

# Example for Remove Nth from End
head2 = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
n2 = 2
print(f"2. Removing {n2}th node from end of 1->2->3->4->5: ", end="")
new_head2 = remove_nth_from_end(head2, n2)
print_list(new_head2)

# =================================================================
# IV. Stacks and Queues
# =================================================================

print("\nIV. Stacks and Queues")
print("-" * 20)

# 1. Implement two stacks in a single array
class TwoStacks:
    """
    Implements two stacks in a single array.
    """
    def __init__(self, n):
        self.size = n
        self.arr = [None] * n
        self.top1 = -1
        self.top2 = self.size

    def push1(self, x):
        if self.top1 < self.top2 - 1:
            self.top1 += 1
            self.arr[self.top1] = x
        else:
            print("Stack Overflow for stack 1")

    def push2(self, x):
        if self.top1 < self.top2 - 1:
            self.top2 -= 1
            self.arr[self.top2] = x
        else:
            print("Stack Overflow for stack 2")

    def pop1(self):
        if self.top1 >= 0:
            x = self.arr[self.top1]
            self.top1 -= 1
            return x
        else:
            print("Stack Underflow for stack 1")
            return None

    def pop2(self):
        if self.top2 < self.size:
            x = self.arr[self.top2]
            self.top2 += 1
            return x
        else:
            print("Stack Underflow for stack 2")
            return None

# Example for Two Stacks
ts = TwoStacks(5)
ts.push1(5)
ts.push2(10)
ts.push2(15)
ts.push1(11)
ts.push2(7)
print(f"1. Popped from stack1: {ts.pop1()}") # 11
print(f"1. Popped from stack2: {ts.pop2()}") # 7

# 2. Next Greater Element
def next_greater_element(arr: List[int]) -> List[int]:
    """
    Finds the next greater element for each element in an array.
    """
    n = len(arr)
    result = [-1] * n
    stack = []
    
    for i in range(n - 1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(arr[i])
        
    return result

# Example for Next Greater Element
arr_nge = [4, 5, 2, 25]
print(f"2. Next greater elements for {arr_nge}: {next_greater_element(arr_nge)}") # Expected: [5, 25, 25, -1]

# 3. Stack Using Queues
class StackUsingQueues:
    """
    Implements a stack using two queues.
    """
    def __init__(self):
        self.q1 = collections.deque()
        self.q2 = collections.deque()

    def push(self, x):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        if self.q1:
            return self.q1.popleft()
        return None

    def top(self):
        if self.q1:
            return self.q1[0]
        return None

    def empty(self):
        return not self.q1

# Example for Stack Using Queues
s_q = StackUsingQueues()
s_q.push(1)
s_q.push(2)
print(f"3. Top of stack using queues: {s_q.top()}") # 2
print(f"3. Popped from stack: {s_q.pop()}") # 2
print(f"3. Is stack empty? {s_q.empty()}") # False

# =================================================================
# V. Bit Manipulation
# =================================================================

print("\nV. Bit Manipulation")
print("-" * 20)

# 1. Power of Two
def is_power_of_two(n: int) -> bool:
    """
    Checks if a number is a power of two using bit manipulation.
    """
    return n > 0 and (n & (n - 1)) == 0

# Example for Power of Two
n1 = 16
print(f"1. Is {n1} a power of two? {is_power_of_two(n1)}") # True
n1_2 = 18
print(f"1. Is {n1_2} a power of two? {is_power_of_two(n1_2)}") # False

# 2. Counting Bits
def count_bits(n: int) -> List[int]:
    """
    Counts the number of 1s in the binary representation of numbers from 0 to n.
    """
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans

# Example for Counting Bits
n2 = 5
print(f"2. Counting bits up to {n2}: {count_bits(n2)}") # [0, 1, 1, 2, 1, 2]

# 3. Maximum XOR of Two Numbers in an Array
def find_maximum_xor(nums: List[int]) -> int:
    """
    Finds the maximum XOR of two numbers in an array using a Trie.
    """
    trie = {}
    max_xor = 0
    L = len(bin(max(nums))) - 2 # Length of max number in binary

    for num in nums:
        # Build Trie
        node = trie
        for i in range(L - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
        
        # Find max XOR for current num
        node = trie
        current_xor = 0
        for i in range(L - 1, -1, -1):
            bit = (num >> i) & 1
            opposite_bit = 1 - bit
            if opposite_bit in node:
                current_xor |= (1 << i)
                node = node[opposite_bit]
            else:
                node = node[bit]
        max_xor = max(max_xor, current_xor)
        
    return max_xor

# Example for Max XOR
arr_xor = [3, 10, 5, 25, 2, 8]
print(f"3. Maximum XOR in {arr_xor}: {find_maximum_xor(arr_xor)}") # Expected: 28 (5 ^ 25)

# =================================================================
# VI. Trees
# =================================================================

print("\nVI. Trees")
print("-" * 20)

class TreeNode:
    """
    Definition for a binary tree node.
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1. Level Order Traversal
def level_order_traversal(root: TreeNode) -> List[List[int]]:
    """
    Performs a level-order traversal of a binary tree.
    """
    if not root:
        return []
    
    result = []
    queue = collections.deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
        
    return result

# Example for Level Order Traversal
root1 = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(f"1. Level order traversal: {level_order_traversal(root1)}")

# =================================================================
# VII. Graphs
# =================================================================

print("\nVII. Graphs")
print("-" * 20)

# 1. Number of Connected Components
def count_components(n: int, edges: List[List[int]]) -> int:
    """
    Finds the number of connected components in an undirected graph.
    """
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = set()
    count = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
                
    for i in range(n):
        if i not in visited:
            dfs(i)
            count += 1
            
    return count

# Example for Connected Components
n_comp = 5
edges_comp = [[0, 1], [1, 2], [3, 4]]
print(f"1. Number of connected components: {count_components(n_comp, edges_comp)}") # Expected: 2

# 2. Cheapest Flights Within K Stops
def find_cheapest_price(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Finds the cheapest flight route with at most k stops.
    """
    prices = [float('inf')] * n
    prices[src] = 0
    
    for _ in range(k + 1):
        temp_prices = prices.copy()
        for s, d, p in flights:
            if prices[s] == float('inf'):
                continue
            if prices[s] + p < temp_prices[d]:
                temp_prices[d] = prices[s] + p
        prices = temp_prices
        
    return prices[dst] if prices[dst] != float('inf') else -1

# Example for Cheapest Flights
n_flights = 3
flights_data = [[0,1,100],[1,2,100],[0,2,500]]
src_f, dst_f, k_f = 0, 2, 1
print(f"2. Cheapest flight from {src_f} to {dst_f} with {k_f} stops: {find_cheapest_price(n_flights, flights_data, src_f, dst_f, k_f)}") # Expected: 200

# =================================================================
# VIII. Backtracking
# =================================================================

print("\nVIII. Backtracking")
print("-" * 20)

# 1. Permutations
def permute(nums: List[int]) -> List[List[int]]:
    """
    Generates all permutations of a given array.
    """
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start] # backtrack
            
    backtrack(0)
    return result

# Example for Permutations
nums_perm = [1, 2, 3]
print(f"1. Permutations of {nums_perm}: {permute(nums_perm)}")

# 2. Combination Sum
def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Finds all unique combinations of numbers that sum to a target.
    """
    result = []
    
    def backtrack(remaining, combo, start):
        if remaining == 0:
            result.append(list(combo))
            return
        if remaining < 0:
            return
            
        for i in range(start, len(candidates)):
            combo.append(candidates[i])
            backtrack(remaining - candidates[i], combo, i)
            combo.pop() # backtrack
            
    backtrack(target, [], 0)
    return result

# Example for Combination Sum
cands_comb = [2, 3, 6, 7]
target_comb = 7
print(f"2. Combination sum for {cands_comb} with target {target_comb}: {combination_sum(cands_comb, target_comb)}")

# 3. Subsets
def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generates all subsets of a given array.
    """
    result = []
    
    def backtrack(start, current_subset):
        result.append(list(current_subset))
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop() # backtrack
            
    backtrack(0, [])
    return result

# Example for Subsets
nums_sub = [1, 2, 3]
print(f"3. Subsets of {nums_sub}: {subsets(nums_sub)}")

# =================================================================
# IX. Divide and Conquer
# =================================================================

print("\nIX. Divide and Conquer")
print("-" * 20)

# 1. Count Inversions in an Array
def count_inversions(arr: List[int]) -> int:
    """
    Counts the number of inversions in an array using a modified merge sort.
    """
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_merge = merge(left, right)
        
        return merged, inv_left + inv_right + inv_merge

    def merge(left, right):
        merged = []
        inversions = 0
        i, j = 0, 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += (len(left) - i)
                
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    _, total_inversions = merge_sort(arr)
    return total_inversions

# Example for Count Inversions
arr_inv = [8, 4, 2, 1]
print(f"1. Inversions in {arr_inv}: {count_inversions(arr_inv)}") # Expected: 6

# 2. Closest Pair of Points
# This is a complex algorithm, a simplified version is shown for brevity.
# A full implementation is more involved.
def closest_pair(points: List[List[int]]) -> float:
    """
    Finds the closest pair of points in a 2D plane.
    """
    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def brute_force(pts):
        min_dist = float('inf')
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                min_dist = min(min_dist, distance(pts[i], pts[j]))
        return min_dist

    def closest_util(px):
        n = len(px)
        if n <= 3:
            return brute_force(px)
        
        mid = n // 2
        mid_point = px[mid]
        
        dl = closest_util(px[:mid])
        dr = closest_util(px[mid:])
        d = min(dl, dr)
        
        strip = [p for p in px if abs(p[0] - mid_point[0]) < d]
        
        # This part can be optimized
        strip.sort(key=lambda p: p[1])
        min_strip_dist = d
        for i in range(len(strip)):
            for j in range(i + 1, len(strip)):
                if (strip[j][1] - strip[i][1]) >= min_strip_dist:
                    break
                min_strip_dist = min(min_strip_dist, distance(strip[i], strip[j]))
        
        return min(d, min_strip_dist)

    points.sort(key=lambda p: p[0])
    return closest_util(points)

# Example for Closest Pair
points_cp = [[0, 0], [1, 1], [10, 10], [10.5, 10.5]]
print(f"2. Closest pair distance in {points_cp}: {closest_pair(points_cp):.4f}") # Expected: 0.7071

# =================================================================
# X. Dynamic Programming
# =================================================================

print("\nX. Dynamic Programming")
print("-" * 20)

# 1. Climbing Stairs
def climb_stairs(n: int) -> int:
    """
    Finds the number of ways to reach the nth step.
    """
    if n <= 2:
        return n
    one_step_before, two_steps_before = 2, 1
    for _ in range(3, n + 1):
        current = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = current
    return one_step_before

# Example for Climbing Stairs
n_stairs = 4
print(f"1. Ways to climb {n_stairs} stairs: {climb_stairs(n_stairs)}") # Expected: 5

# 2. Coin Change (Minimum Coins)
def coin_change(coins: List[int], amount: int) -> int:
    """
    Finds the minimum number of coins needed to make the amount.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
            
    return dp[amount] if dp[amount] != float('inf') else -1

# Example for Coin Change
coins_cc = [1, 2, 5]
amount_cc = 11
print(f"2. Min coins for amount {amount_cc} with {coins_cc}: {coin_change(coins_cc, amount_cc)}") # Expected: 3

# 3. Longest Increasing Subsequence (LIS)
def length_of_lis(nums: List[int]) -> int:
    """
    Finds the length of the longest increasing subsequence.
    """
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Example for LIS
nums_lis = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"3. LIS of {nums_lis}: {length_of_lis(nums_lis)}") # Expected: 4

# 4. Longest Palindromic Subsequence
def longest_palindromic_subsequence(s: str) -> int:
    """
    Finds the length of the longest palindromic subsequence.
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i+1][j-1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
                
    return dp[0][n-1]

# Example for Longest Palindromic Subsequence
s_lps = "bbbab"
print(f"4. Longest palindromic subsequence of '{s_lps}': {longest_palindromic_subsequence(s_lps)}") # Expected: 4

# 5. House Robber
def house_robber(nums: List[int]) -> int:
    """
    Finds the maximum amount of money you can rob without robbing adjacent houses.
    """
    rob1, rob2 = 0, 0
    # [rob1, rob2, n, n+1, ...]
    for n in nums:
        temp = max(n + rob1, rob2)
        rob1 = rob2
        rob2 = temp
    return rob2

# Example for House Robber
nums_hr = [2, 7, 9, 3, 1]
print(f"5. Max amount to rob from {nums_hr}: {house_robber(nums_hr)}") # Expected: 12

# 6. Minimum Cost to Cut a Stick
def min_cost_cut_stick(n: int, cuts: List[int]) -> int:
    """
    Finds the minimum cost to cut a stick into pieces.
    """
    memo = {}
    cuts = sorted([0] + cuts + [n])
    
    def dp(i, j):
        if j - i <= 1:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]
        
        res = float('inf')
        for k in range(i + 1, j):
            res = min(res, dp(i, k) + dp(k, j))
            
        memo[(i, j)] = res + (cuts[j] - cuts[i])
        return memo[(i, j)]
        
    return dp(0, len(cuts) - 1)

# Example for Min Cost to Cut Stick
n_stick = 7
cuts_stick = [1, 3, 4, 5]
print(f"6. Min cost to cut stick of length {n_stick} at {cuts_stick}: {min_cost_cut_stick(n_stick, cuts_stick)}") # Expected: 16

# =================================================================
# XI. Greedy Algorithms
# =================================================================

print("\nXI. Greedy Algorithms")
print("-" * 20)

# 1. Jump Game II
def jump_game_ii(nums: List[int]) -> int:
    """
    Finds the minimum number of jumps to reach the last index.
    """
    jumps = 0
    current_end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

# Example for Jump Game II
nums_jg = [2, 3, 1, 1, 4]
print(f"1. Min jumps for {nums_jg}: {jump_game_ii(nums_jg)}") # Expected: 2

# 2. Assign Cookies
def assign_cookies(g: List[int], s: List[int]) -> int:
    """
    Assigns cookies to children to maximize satisfied children.
    """
    g.sort()
    s.sort()
    child_i, cookie_j = 0, 0
    while child_i < len(g) and cookie_j < len(s):
        if s[cookie_j] >= g[child_i]:
            child_i += 1
        cookie_j += 1
    return child_i

# Example for Assign Cookies
g_cookies = [1, 2, 3]
s_cookies = [1, 1]
print(f"2. Max satisfied children: {assign_cookies(g_cookies, s_cookies)}") # Expected: 1

# 3. Huffman Encoding
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text: str):
    """
    Builds an optimal prefix code for characters based on their frequencies.
    Returns the codes as a dictionary.
    """
    if not text:
        return {}
        
    frequency = collections.Counter(text)
    priority_queue = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(priority_queue, merged)
        
    root = priority_queue[0]
    codes = {}
    
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
            return
        
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        
    generate_codes(root, "")
    return codes

# Example for Huffman Encoding
text_huffman = "aabcddd"
print(f"3. Huffman codes for '{text_huffman}': {huffman_encoding(text_huffman)}")
