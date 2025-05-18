import collections
import heapq
import itertools
import math
from typing import List, Optional, Tuple

# Helper Node class for Linked List problems
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        vals = []
        curr = self
        while curr:
            vals.append(str(curr.val))
            curr = curr.next
        return " -> ".join(vals)

# --- Topic 1 & 4: Prefix Sum Array ---
def calculate_prefix_sum(nums: List[int]) -> List[int]:
    """Calculates the prefix sum array."""
    prefix_sum = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix_sum[i + 1] = prefix_sum[i] + nums[i]
    return prefix_sum

def range_sum_query(prefix_sum: List[int], left: int, right: int) -> int:
    """
    Finds the sum of elements in the range [left, right] (inclusive)
    using a precomputed prefix sum array.
    Assumes 0-based indexing for the original array.
    """
    if left < 0 or right >= len(prefix_sum) - 1 or left > right:
        raise ValueError("Invalid range")
    # prefix_sum[right + 1] is sum up to index right
    # prefix_sum[left] is sum up to index left - 1
    return prefix_sum[right + 1] - prefix_sum[left]

# --- Topic 2: Equilibrium Index ---
def find_equilibrium_index(nums: List[int]) -> int:
    """
    Finds an index in an array such that the sum of elements to the left
    is equal to the sum of elements to the right. Returns -1 if no such index exists.
    """
    total_sum = sum(nums)
    left_sum = 0
    for i in range(len(nums)):
        # right_sum = total_sum - left_sum - nums[i]
        if left_sum == total_sum - left_sum - nums[i]:
            return i
        left_sum += nums[i]
    return -1

# --- Topic 3 & 5: Split Array into Equal Sum Prefix and Suffix ---
def can_split_equal_sum(nums: List[int]) -> bool:
    """
    Checks if an array can be split into two non-empty parts (prefix and suffix)
    such that the sum of the prefix equals the sum of the suffix.
    """
    total_sum = sum(nums)
    left_sum = 0
    # We need non-empty prefix and suffix, so iterate up to len(nums) - 1
    for i in range(len(nums) - 1):
        left_sum += nums[i]
        right_sum = total_sum - left_sum
        if left_sum == right_sum:
            return True
    return False

# --- Topic 4: Maximum Subarray of Size K ---
def max_subarray_sum_size_k(nums: List[int], k: int) -> int:
    """Finds the maximum sum of any contiguous subarray of size K."""
    if k <= 0 or k > len(nums):
        return 0 # Or raise error

    max_sum = -float('inf')
    current_sum = 0

    # Calculate sum of the first window
    for i in range(k):
        current_sum += nums[i]
    max_sum = current_sum

    # Slide the window
    for i in range(k, len(nums)):
        current_sum += nums[i] - nums[i - k] # Add new element, remove old one
        max_sum = max(max_sum, current_sum)

    return max_sum if max_sum != -float('inf') else 0 # Handle case where all numbers are negative


# --- Topic 5: Longest Substring Without Repeating Characters ---
def length_of_longest_substring(s: str) -> int:
    """Finds the length of the longest substring without repeating characters."""
    char_set = set()
    left = 0
    max_length = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    return max_length

# --- Topic 6: Find two numbers in a sorted array that add up to a target ---
def two_sum_sorted(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Finds two indices in a sorted array whose elements add up to the target.
    Uses the two-pointer technique. Returns indices (1-based) or None.
    """
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return (left + 1, right + 1) # Or 0-based: (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None

# --- Topic 7 & 9: Majority Element ( > n/2 times) ---
def majority_element_boyer_moore(nums: List[int]) -> int:
    """
    Finds the element that appears more than n/2 times using Boyer-Moore Voting Algorithm.
    Assumes a majority element always exists.
    """
    candidate = None
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1
    # Optional: Verify if the candidate is actually the majority element
    # count = 0
    # for num in nums:
    #     if num == candidate:
    #         count += 1
    # if count > len(nums) // 2:
    #     return candidate
    # else:
    #     return -1 # Or raise error if majority element might not exist
    return candidate

# --- Topic 8 & 10: Next Permutation ---
def next_permutation(nums: List[int]) -> None:
    """
    Rearranges numbers into the lexicographically next greater permutation in-place.
    If such an arrangement is not possible, it rearranges to the lowest possible order.
    """
    n = len(nums)
    # Step 1: Find the largest index k such that nums[k] < nums[k + 1]
    k = n - 2
    while k >= 0 and nums[k] >= nums[k + 1]:
        k -= 1

    if k == -1:
        # If no such index exists, reverse the whole array
        nums.reverse()
        return

    # Step 2: Find the largest index l > k such that nums[k] < nums[l]
    l = n - 1
    while nums[l] <= nums[k]:
        l -= 1

    # Step 3: Swap nums[k] and nums[l]
    nums[k], nums[l] = nums[l], nums[k]

    # Step 4: Reverse the sub-array nums[k + 1:]
    left, right = k + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

# --- Topic 9 & 23: Sliding Window Maximum (using deque) ---
def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """Finds the maximum value in every sliding window of size K using a deque."""
    if not nums or k == 0:
        return []
    if k >= len(nums):
        return [max(nums)]

    result = []
    # Deque stores indices of elements, maintaining decreasing order of values
    dq = collections.deque()

    for i in range(len(nums)):
        # Remove elements out of the current window from the front of the deque
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove elements smaller than the current element from the back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # The front of the deque is the max for the window ending at i
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# --- Topic 10: Maximum Subarray (Kadane's Algorithm) ---
def max_subarray_kadane(nums: List[int]) -> int:
    """Finds the contiguous subarray with the largest sum using Kadane's algorithm."""
    max_so_far = -float('inf')
    current_max = 0
    for num in nums:
        current_max += num
        if current_max > max_so_far:
            max_so_far = current_max
        if current_max < 0:
            current_max = 0 # Reset if the sum becomes negative
    # Handle case where all numbers are negative
    if max_so_far == -float('inf'):
         return max(nums) if nums else 0
    return max_so_far


# --- Topic 11 & 15: Trapping Rainwater ---
def trap_rainwater(height: List[int]) -> int:
    """Calculates how much water can be trapped between the bars of a histogram."""
    if not height:
        return 0

    n = len(height)
    left, right = 0, n - 1
    left_max, right_max = 0, 0
    water_trapped = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water_trapped += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water_trapped += right_max - height[right]
            right -= 1

    return water_trapped

# --- Topic 12: Counting Bits ---
def count_bits(n: int) -> List[int]:
    """Counts the number of 1s in the binary representation of numbers from 0 to n."""
    # dp[i] will store the number of set bits in i
    dp = [0] * (n + 1)
    offset = 1 # Represents the nearest power of 2 less than or equal to i
    for i in range(1, n + 1):
        if offset * 2 == i:
            offset = i
        # The number of set bits in i is 1 + number of set bits in (i - offset)
        # Example: bits(7) = bits(111) = 1 + bits(3) = 1 + bits(011)
        #          bits(6) = bits(110) = 1 + bits(2) = 1 + bits(010)
        dp[i] = 1 + dp[i - offset]
    return dp

# --- Topic 12: Power of Two ---
def is_power_of_two(n: int) -> bool:
    """Checks if a number is a power of two using bit manipulation."""
    # Powers of two have exactly one bit set in their binary representation.
    # n > 0 ensures we handle n=0 correctly.
    # n & (n - 1) clears the least significant set bit. If n is a power of two,
    # this result will be 0.
    return n > 0 and (n & (n - 1) == 0)

# --- Topic 13: Maximum XOR of Two Numbers in an Array ---
def find_maximum_xor(nums: List[int]) -> int:
    """Finds the maximum XOR of two numbers in an array using a Trie."""
    max_xor = 0
    # Determine the maximum number of bits needed
    max_num = max(nums) if nums else 0
    L = len(bin(max_num)) - 2 if max_num > 0 else 1 # Length of binary representation

    trie = {}
    for num in nums:
        node = trie
        # Build the trie path for the current number
        for i in range(L - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

        # Find the best XOR partner for the current number
        node = trie
        current_xor = 0
        for i in range(L - 1, -1, -1):
            bit = (num >> i) & 1
            # We want the opposite bit to maximize XOR
            toggle_bit = 1 - bit
            if toggle_bit in node:
                current_xor |= (1 << i) # Add 2^i to current_xor
                node = node[toggle_bit]
            elif bit in node: # If opposite bit not available, take the same bit
                 node = node[bit]
            else: # Should not happen if trie is built correctly for at least one number
                break
        max_xor = max(max_xor, current_xor)

    return max_xor


# --- Topic 14: Maximum Product Subarray ---
def max_product_subarray(nums: List[int]) -> int:
    """Finds the contiguous subarray with the largest product."""
    if not nums:
        return 0

    max_prod = nums[0]
    min_so_far = nums[0] # Keep track of min product ending here (for negative numbers)
    max_so_far = nums[0] # Keep track of max product ending here

    for i in range(1, len(nums)):
        num = nums[i]
        # When we multiply by a negative number, max becomes min and min becomes max
        if num < 0:
            min_so_far, max_so_far = max_so_far, min_so_far

        # Update min and max product ending at index i
        # Either start a new subarray with num or extend the previous one
        min_so_far = min(num, min_so_far * num)
        max_so_far = max(num, max_so_far * num)

        # Update the overall maximum product found so far
        max_prod = max(max_prod, max_so_far)

    return max_prod


# --- Topic 14: Count Numbers with Unique Digits ---
def count_numbers_with_unique_digits(n: int) -> int:
    """Counts all non-negative integers x with unique digits, where 0 <= x < 10^n."""
    if n == 0:
        return 1 # Only 0

    # Count for 1-digit numbers (0-9)
    count = 10
    # Available digits for the next position
    available_digits = 9
    # Unique digits for the current length
    unique_digits_k = 9

    # Calculate for lengths 2 to n
    for k in range(2, min(n + 1, 11)): # Max 10 unique digits (0-9)
        # For k-digit numbers:
        # First digit: 9 choices (1-9)
        # Second digit: 9 choices (0-9 excluding first)
        # Third digit: 8 choices ...
        unique_digits_k *= available_digits
        count += unique_digits_k
        available_digits -= 1

    return count

# --- Topic 16 & 20: Longest Palindromic Substring ---
def longest_palindrome(s: str) -> str:
    """Finds the longest palindromic substring in a given string."""
    if not s or len(s) < 1:
        return ""

    start, end = 0, 0 # Indices of the longest palindrome found so far

    def expand_around_center(left, right):
        # Expand while the characters match and indices are within bounds
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # Return the length of the palindrome found
        # The actual palindrome is s[left+1:right]
        return right - left - 1

    for i in range(len(s)):
        # Check for odd length palindromes (center is i)
        len1 = expand_around_center(i, i)
        # Check for even length palindromes (center is between i and i+1)
        len2 = expand_around_center(i, i + 1)

        max_len = max(len1, len2)

        # Update start and end if a longer palindrome is found
        if max_len > (end - start):
            # Calculate the start index based on the center and length
            start = i - (max_len - 1) // 2
            end = i + max_len // 2

    return s[start : end + 1]


# --- Topic 17 & 20: Longest Common Prefix ---
def longest_common_prefix(strs: List[str]) -> str:
    """Finds the longest common prefix string amongst an array of strings."""
    if not strs:
        return ""

    # Sort the strings lexicographically. The LCP will be the common prefix
    # between the first and the last string in the sorted list.
    strs.sort()
    first_str = strs[0]
    last_str = strs[-1]
    lcp = []

    for i in range(len(first_str)):
        if i < len(last_str) and first_str[i] == last_str[i]:
            lcp.append(first_str[i])
        else:
            break # Mismatch found

    return "".join(lcp)

# --- Topic 18: Merge Two Sorted Linked Lists ---
def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """Merges two sorted linked lists into one sorted list."""
    dummy = ListNode() # Dummy head for the merged list
    current = dummy

    p1, p2 = list1, list2

    while p1 and p2:
        if p1.val <= p2.val:
            current.next = p1
            p1 = p1.next
        else:
            current.next = p2
            p2 = p2.next
        current = current.next

    # Attach the remaining part of the non-empty list
    if p1:
        current.next = p1
    elif p2:
        current.next = p2

    return dummy.next

# --- Topic 18: Remove N-th Node From End of List ---
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """Removes the n-th node from the end of a singly linked list."""
    dummy = ListNode(0, head) # Dummy node to handle edge case (removing head)
    fast = dummy
    slow = dummy

    # Advance fast pointer n steps ahead
    for _ in range(n):
        if fast.next:
             fast = fast.next
        else:
            return head # n is larger than list size, invalid

    # Move both pointers until fast reaches the end
    while fast.next:
        fast = fast.next
        slow = slow.next

    # slow is now at the node *before* the one to be removed
    slow.next = slow.next.next # Skip the nth node from the end

    return dummy.next


# --- Topic 19: Palindrome Number ---
def is_palindrome_number(x: int) -> bool:
    """Checks if an integer is a palindrome without converting it to a string."""
    # Negative numbers are not palindromes
    # Numbers ending in 0 (except 0 itself) cannot be palindromes
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    reverted_number = 0
    while x > reverted_number:
        reverted_number = reverted_number * 10 + x % 10
        x //= 10

    # When the length is an odd number, we can get rid of the middle digit by reverted_number // 10
    # For example when the input is 12321, at the end of the while loop we get x = 12, reverted_number = 123,
    # since the middle digit doesn't matter in palidrome(it will always equal to itself), we can simply get rid of it.
    return x == reverted_number or x == reverted_number // 10


# --- Topic 21: Intersection of Two Linked Lists ---
def get_intersection_node(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """Finds the node where two singly linked lists intersect."""
    if not headA or not headB:
        return None

    pA = headA
    pB = headB

    # Traverse both lists. If a pointer reaches the end, switch it to the
    # head of the other list. If they intersect, they will meet.
    # If they don't intersect, both will become None eventually.
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA # Either the intersection node or None


# --- Topic 22: Implement Two Stacks in a Single Array ---
class TwoStacks:
    def __init__(self, capacity):
        if capacity < 2:
            raise ValueError("Capacity must be at least 2")
        self.capacity = capacity
        self.arr = [None] * capacity
        self.top1 = -1          # Top of stack 1 (grows from left)
        self.top2 = capacity    # Top of stack 2 (grows from right)

    def push1(self, value):
        """Pushes element onto stack 1."""
        # Check for overflow: Is there space between top1 and top2?
        if self.top1 < self.top2 - 1:
            self.top1 += 1
            self.arr[self.top1] = value
        else:
            print("Stack Overflow for Stack 1")
            # Or raise an exception

    def push2(self, value):
        """Pushes element onto stack 2."""
        # Check for overflow
        if self.top1 < self.top2 - 1:
            self.top2 -= 1
            self.arr[self.top2] = value
        else:
            print("Stack Overflow for Stack 2")
            # Or raise an exception

    def pop1(self):
        """Pops element from stack 1."""
        if self.top1 >= 0:
            value = self.arr[self.top1]
            self.arr[self.top1] = None # Optional cleanup
            self.top1 -= 1
            return value
        else:
            print("Stack Underflow for Stack 1")
            return None # Or raise an exception

    def pop2(self):
        """Pops element from stack 2."""
        if self.top2 < self.capacity:
            value = self.arr[self.top2]
            self.arr[self.top2] = None # Optional cleanup
            self.top2 += 1
            return value
        else:
            print("Stack Underflow for Stack 2")
            return None # Or raise an exception

    def peek1(self):
        """Returns the top element of stack 1 without removing it."""
        if self.top1 >= 0:
            return self.arr[self.top1]
        return None

    def peek2(self):
        """Returns the top element of stack 2 without removing it."""
        if self.top2 < self.capacity:
            return self.arr[self.top2]
        return None

    def is_empty1(self):
        return self.top1 == -1

    def is_empty2(self):
        return self.top2 == self.capacity

# --- Topic 22: Next Greater Element ---
def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Finds the next greater element for each element in nums1 within nums2.
    Uses a monotonic stack.
    """
    next_greater_map = {} # Map num to its next greater element in nums2
    stack = [] # Monotonic decreasing stack

    # Iterate through nums2 to find next greater elements
    for num in nums2:
        # While stack is not empty and current num is greater than stack top
        while stack and num > stack[-1]:
            smaller_num = stack.pop()
            next_greater_map[smaller_num] = num
        stack.append(num)

    # For elements remaining in the stack, there's no greater element
    while stack:
        next_greater_map[stack.pop()] = -1

    # Build the result for nums1 using the map
    result = [next_greater_map.get(num, -1) for num in nums1]
    return result

# --- Topic 24: Largest Rectangle in Histogram ---
def largest_rectangle_area(heights: List[int]) -> int:
    """Finds the largest rectangle that can be formed in a histogram."""
    stack = [-1] # Stack stores indices, -1 helps calculate width for bars reaching start
    max_area = 0
    for i, h in enumerate(heights):
        # While stack top bar is taller than or equal to current bar h
        while stack[-1] != -1 and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1 # Width is distance between current index and index before the popped bar
            max_area = max(max_area, height * width)
        stack.append(i) # Push current index onto stack

    # Process remaining bars in the stack
    while stack[-1] != -1:
        height = heights[stack.pop()]
        width = len(heights) - stack[-1] - 1 # Width extends to the end
        max_area = max(max_area, height * width)

    return max_area

# --- Topic 26 & 29: Permutations ---
def generate_permutations(nums: List[int]) -> List[List[int]]:
    """Generates all permutations of a given array using backtracking."""
    result = []
    n = len(nums)

    def backtrack(start):
        if start == n:
            result.append(nums[:]) # Append a copy
            return

        for i in range(start, n):
            # Swap current element with the start element
            nums[start], nums[i] = nums[i], nums[start]
            # Recurse for the rest of the array
            backtrack(start + 1)
            # Backtrack: swap back to restore original order for next iteration
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result

# --- Topic 26 & 29: Subsets ---
def generate_subsets(nums: List[int]) -> List[List[int]]:
    """Generates all subsets (the power set) of a given array using backtracking."""
    result = []
    n = len(nums)

    def backtrack(start, current_subset):
        # Add the current subset to the result
        result.append(current_subset[:]) # Append a copy

        # Explore adding more elements
        for i in range(start, n):
            # Include nums[i] in the current subset
            current_subset.append(nums[i])
            # Recurse
            backtrack(i + 1, current_subset)
            # Backtrack: remove nums[i] to explore subsets without it
            current_subset.pop()

    backtrack(0, [])
    return result

# --- Topic 27: Combination Sum ---
def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Finds all unique combinations of numbers from candidates that sum to target.
    Numbers can be reused.
    """
    result = []
    n = len(candidates)
    # Sorting helps avoid duplicate combinations if candidates list has duplicates
    # and allows for pruning the search space.
    # candidates.sort() # Uncomment if candidates might have duplicates and only unique combinations are needed

    def backtrack(start_index, current_combination, current_sum):
        if current_sum == target:
            result.append(current_combination[:]) # Append a copy
            return
        if current_sum > target:
            return # Pruning

        for i in range(start_index, n):
            # Include candidates[i]
            current_combination.append(candidates[i])
            # Recurse, allowing reuse of the same number (pass i, not i+1)
            backtrack(i, current_combination, current_sum + candidates[i])
            # Backtrack
            current_combination.pop()

    backtrack(0, [], 0)
    return result


# --- Topic 27 & 30: Find the Element with Maximum Frequency ---
def find_max_frequency_element(nums: List[int]) -> Optional[int]:
    """Finds the element that appears most frequently in an array."""
    if not nums:
        return None
    counts = collections.Counter(nums)
    # Find the element with the maximum count
    max_freq_element = max(counts, key=counts.get)
    # Alternatively, to get both element and frequency:
    # max_freq_element, max_count = counts.most_common(1)[0]
    return max_freq_element

# --- Topic 28 & 31: Median of Two Sorted Arrays ---
def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """Finds the median of two sorted arrays."""
    n1, n2 = len(nums1), len(nums2)
    # Ensure nums1 is the shorter array for efficiency
    if n1 > n2:
        return find_median_sorted_arrays(nums2, nums1)

    n = n1 + n2
    left, right = 0, n1 # Binary search range for partition in nums1

    while left <= right:
        partition1 = (left + right) // 2 # Partition index for nums1
        partition2 = (n + 1) // 2 - partition1 # Corresponding partition index for nums2

        # Get elements around the partitions
        max_left1 = nums1[partition1 - 1] if partition1 > 0 else -float('inf')
        min_right1 = nums1[partition1] if partition1 < n1 else float('inf')

        max_left2 = nums2[partition2 - 1] if partition2 > 0 else -float('inf')
        min_right2 = nums2[partition2] if partition2 < n2 else float('inf')

        # Check if partitions are correct
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if n % 2 == 0: # Even total length
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else: # Odd total length
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # Partition in nums1 is too far right, move left
            right = partition1 - 1
        else: # max_left2 > min_right1
            # Partition in nums1 is too far left, move right
            left = partition1 + 1
    # Should not be reached if arrays are sorted
    return -1.0


# --- Topic 28: Kth Smallest Element in a Sorted Matrix ---
def kth_smallest_in_matrix(matrix: List[List[int]], k: int) -> int:
    """
    Finds the k-th smallest element in a sorted matrix where each row and column is sorted.
    Note: Syllabus mentioned "from the last", which is ambiguous. This implements
    the standard k-th smallest overall.
    """
    n = len(matrix)
    min_heap = [] # Store tuples (value, row, col)

    # Push the first element of each row into the heap
    for r in range(min(k, n)): # Optimization: only need first k rows potentially
        heapq.heappush(min_heap, (matrix[r][0], r, 0))

    count = 0
    result = -1

    # Extract the smallest element k times
    while min_heap and count < k:
        result, r, c = heapq.heappop(min_heap)
        count += 1

        # If there's a next element in the same row, push it
        if c + 1 < n:
            heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))

    return result


# --- Topic 29 & 33: Top K Frequent Elements ---
def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """Finds the k most frequent elements using a min-heap (priority queue)."""
    if k == len(nums):
        return nums

    # 1. Count frequencies
    counts = collections.Counter(nums)

    # 2. Use a min-heap to keep track of the top k frequent elements
    # Heap stores (frequency, element) tuples
    min_heap = []
    for num, freq in counts.items():
        heapq.heappush(min_heap, (freq, num))
        # If heap size exceeds k, remove the element with the smallest frequency
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    # 3. Extract elements from the heap
    top_k = [num for freq, num in min_heap]
    return top_k

# --- Topic 31: Two Sum (using hashing) ---
def two_sum_hash(nums: List[int], target: int) -> Optional[List[int]]:
    """
    Finds two indices in an array (not necessarily sorted) whose elements add up to the target.
    Uses a hash map (dictionary). Returns 0-based indices or None.
    """
    num_map = {} # Stores number -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i # Store the current number and its index
    return None

# --- Topic 32: Subarray Sum Equals K (using hashing) ---
def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Finds the total number of continuous subarrays whose sum equals k.
    Uses hashing with prefix sums.
    """
    count = 0
    current_sum = 0
    # Dictionary to store frequency of prefix sums encountered so far
    prefix_sum_counts = {0: 1} # Initialize with sum 0 having frequency 1

    for num in nums:
        current_sum += num
        # Check if (current_sum - k) exists in the map
        # If it does, it means a subarray ending here with sum k exists
        complement = current_sum - k
        if complement in prefix_sum_counts:
            count += prefix_sum_counts[complement]

        # Update the frequency of the current prefix sum
        prefix_sum_counts[current_sum] = prefix_sum_counts.get(current_sum, 0) + 1

    return count


if __name__ == "__main__":
    print("--- Prefix Sum ---")
    nums_ps = [1, 2, 3, 4, 5]
    prefix_sum = calculate_prefix_sum(nums_ps)
    print(f"Array: {nums_ps}, Prefix Sum: {prefix_sum}")
    print(f"Sum range [1, 3]: {range_sum_query(prefix_sum, 1, 3)}") # 2+3+4 = 9

    print("\n--- Equilibrium Index ---")
    nums_eq = [-7, 1, 5, 2, -4, 3, 0]
    print(f"Array: {nums_eq}, Equilibrium Index: {find_equilibrium_index(nums_eq)}") # Index 3

    print("\n--- Split Equal Sum ---")
    nums_split = [1, 5, 11, 5]
    print(f"Array: {nums_split}, Can split equally? {can_split_equal_sum(nums_split)}") # True (1+5 vs 11+5 -> False, 1+5+11 vs 5 -> False)
    nums_split_2 = [10, 4, -8, 6]
    print(f"Array: {nums_split_2}, Can split equally? {can_split_equal_sum(nums_split_2)}") # True (10+4 = 14, -8+6 = -2 -> False)

    print("\n--- Max Subarray Size K ---")
    nums_msk = [2, 1, 5, 1, 3, 2]
    k_msk = 3
    print(f"Array: {nums_msk}, K={k_msk}, Max Sum: {max_subarray_sum_size_k(nums_msk, k_msk)}") # 9 (5+1+3)

    print("\n--- Longest Substring Without Repeating ---")
    s_lswrc = "abcabcbb"
    print(f"String: '{s_lswrc}', Length: {length_of_longest_substring(s_lswrc)}") # 3 ("abc")

    print("\n--- Two Sum Sorted ---")
    nums_ts_sorted = [2, 7, 11, 15]
    target_ts_sorted = 9
    print(f"Array: {nums_ts_sorted}, Target: {target_ts_sorted}, Indices: {two_sum_sorted(nums_ts_sorted, target_ts_sorted)}") # (1, 2)

    print("\n--- Majority Element ---")
    nums_maj = [2, 2, 1, 1, 1, 2, 2]
    print(f"Array: {nums_maj}, Majority Element: {majority_element_boyer_moore(nums_maj)}") # 2

    print("\n--- Next Permutation ---")
    nums_np = [1, 2, 3]
    print(f"Original: {nums_np}", end=" -> ")
    next_permutation(nums_np)
    print(f"Next: {nums_np}") # [1, 3, 2]
    nums_np = [3, 2, 1]
    print(f"Original: {nums_np}", end=" -> ")
    next_permutation(nums_np)
    print(f"Next: {nums_np}") # [1, 2, 3]

    print("\n--- Sliding Window Maximum ---")
    nums_swm = [1, 3, -1, -3, 5, 3, 6, 7]
    k_swm = 3
    print(f"Array: {nums_swm}, K={k_swm}, Maxima: {sliding_window_maximum(nums_swm, k_swm)}") # [3, 3, 5, 5, 6, 7]

    print("\n--- Max Subarray (Kadane) ---")
    nums_kadane = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Array: {nums_kadane}, Max Sum: {max_subarray_kadane(nums_kadane)}") # 6 ([4, -1, 2, 1])

    print("\n--- Trapping Rainwater ---")
    height_tr = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(f"Heights: {height_tr}, Water Trapped: {trap_rainwater(height_tr)}") # 6

    print("\n--- Counting Bits ---")
    n_cb = 5
    print(f"Count bits up to {n_cb}: {count_bits(n_cb)}") # [0, 1, 1, 2, 1, 2]

    print("\n--- Power of Two ---")
    print(f"Is 16 power of two? {is_power_of_two(16)}") # True
    print(f"Is 18 power of two? {is_power_of_two(18)}") # False

    print("\n--- Max XOR ---")
    nums_xor = [3, 10, 5, 25, 2, 8]
    print(f"Array: {nums_xor}, Max XOR: {find_maximum_xor(nums_xor)}") # 28 (5 ^ 25)

    print("\n--- Max Product Subarray ---")
    nums_mps = [2, 3, -2, 4]
    print(f"Array: {nums_mps}, Max Product: {max_product_subarray(nums_mps)}") # 6 ([2, 3])
    nums_mps2 = [-2, 0, -1]
    print(f"Array: {nums_mps2}, Max Product: {max_product_subarray(nums_mps2)}") # 0

    print("\n--- Count Unique Digits ---")
    n_cud = 2
    print(f"Count unique digits for n={n_cud}: {count_numbers_with_unique_digits(n_cud)}") # 91 (10 for 1-digit + 81 for 2-digits)

    print("\n--- Longest Palindromic Substring ---")
    s_lps = "babad"
    print(f"String: '{s_lps}', Longest Palindrome: {longest_palindrome(s_lps)}") # "bab" or "aba"

    print("\n--- Longest Common Prefix ---")
    strs_lcp = ["flower", "flow", "flight"]
    print(f"Strings: {strs_lcp}, LCP: {longest_common_prefix(strs_lcp)}") # "fl"

    print("\n--- Merge Two Sorted Lists ---")
    l1 = ListNode(1, ListNode(2, ListNode(4)))
    l2 = ListNode(1, ListNode(3, ListNode(4)))
    merged = merge_two_lists(l1, l2)
    print(f"Merged List: {merged}") # 1 -> 1 -> 2 -> 3 -> 4 -> 4

    print("\n--- Remove Nth From End ---")
    head_rne = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    n_rne = 2
    print(f"Original: {head_rne}, N={n_rne}", end=" -> ")
    new_head = remove_nth_from_end(head_rne, n_rne)
    print(f"After Remove: {new_head}") # 1 -> 2 -> 3 -> 5

    print("\n--- Palindrome Number ---")
    print(f"Is 121 palindrome? {is_palindrome_number(121)}") # True
    print(f"Is -121 palindrome? {is_palindrome_number(-121)}") # False
    print(f"Is 10 palindrome? {is_palindrome_number(10)}") # False

    print("\n--- Intersection of Linked Lists ---")
    # Create intersection: listA = 4->1->8->4->5, listB = 5->6->1->8->4->5
    common = ListNode(8, ListNode(4, ListNode(5)))
    headA = ListNode(4, ListNode(1, common))
    headB = ListNode(5, ListNode(6, ListNode(1, common)))
    intersect_node = get_intersection_node(headA, headB)
    print(f"Intersection Node Value: {intersect_node.val if intersect_node else None}") # 8

    print("\n--- Two Stacks ---")
    two_stacks = TwoStacks(5)
    two_stacks.push1(1)
    two_stacks.push2(5)
    two_stacks.push1(2)
    two_stacks.push2(4)
    print(f"Pop1: {two_stacks.pop1()}") # 2
    print(f"Pop2: {two_stacks.pop2()}") # 4
    print(f"Peek1: {two_stacks.peek1()}") # 1
    print(f"Is Stack2 Empty? {two_stacks.is_empty2()}") # False

    print("\n--- Next Greater Element ---")
    nums1_nge = [4, 1, 2]
    nums2_nge = [1, 3, 4, 2]
    print(f"Nums1: {nums1_nge}, Nums2: {nums2_nge}, Next Greater: {next_greater_element(nums1_nge, nums2_nge)}") # [-1, 3, -1]

    print("\n--- Largest Rectangle in Histogram ---")
    heights_hist = [2, 1, 5, 6, 2, 3]
    print(f"Heights: {heights_hist}, Largest Area: {largest_rectangle_area(heights_hist)}") # 10

    print("\n--- Permutations ---")
    nums_perm = [1, 2, 3]
    print(f"Array: {nums_perm}, Permutations: {generate_permutations(nums_perm)}")

    print("\n--- Subsets ---")
    nums_subs = [1, 2, 3]
    print(f"Array: {nums_subs}, Subsets: {generate_subsets(nums_subs)}")

    print("\n--- Combination Sum ---")
    cands_cs = [2, 3, 6, 7]
    target_cs = 7
    print(f"Candidates: {cands_cs}, Target: {target_cs}, Combinations: {combination_sum(cands_cs, target_cs)}") # [[2, 2, 3], [7]]

    print("\n--- Max Frequency Element ---")
    nums_mf = [1, 2, 3, 2, 2, 1, 4, 2]
    print(f"Array: {nums_mf}, Max Freq Element: {find_max_frequency_element(nums_mf)}") # 2

    print("\n--- Median of Two Sorted Arrays ---")
    nums1_med = [1, 3]
    nums2_med = [2]
    print(f"Arrays: {nums1_med}, {nums2_med}, Median: {find_median_sorted_arrays(nums1_med, nums2_med)}") # 2.0
    nums1_med2 = [1, 2]
    nums2_med2 = [3, 4]
    print(f"Arrays: {nums1_med2}, {nums2_med2}, Median: {find_median_sorted_arrays(nums1_med2, nums2_med2)}") # 2.5

    print("\n--- Kth Smallest in Matrix ---")
    matrix_ks = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    k_ks = 8
    print(f"Matrix: {matrix_ks}, K={k_ks}, Kth Smallest: {kth_smallest_in_matrix(matrix_ks, k_ks)}") # 13

    print("\n--- Top K Frequent Elements ---")
    nums_tkf = [1, 1, 1, 2, 2, 3]
    k_tkf = 2
    print(f"Array: {nums_tkf}, K={k_tkf}, Top K Frequent: {top_k_frequent(nums_tkf, k_tkf)}") # [1, 2] or [2, 1]

    print("\n--- Two Sum Hash ---")
    nums_ts_hash = [2, 7, 11, 15]
    target_ts_hash = 9
    print(f"Array: {nums_ts_hash}, Target: {target_ts_hash}, Indices: {two_sum_hash(nums_ts_hash, target_ts_hash)}") # [0, 1]

    print("\n--- Subarray Sum Equals K ---")
    nums_ssk = [1, 1, 1]
    k_ssk = 2
    print(f"Array: {nums_ssk}, K={k_ssk}, Count: {subarray_sum_equals_k(nums_ssk, k_ssk)}") # 2
    nums_ssk2 = [1, 2, 3]
    k_ssk2 = 3
    print(f"Array: {nums_ssk2}, K={k_ssk2}, Count: {subarray_sum_equals_k(nums_ssk2, k_ssk2)}") # 2 ([1, 2], [3])

