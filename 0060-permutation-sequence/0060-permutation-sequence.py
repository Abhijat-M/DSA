class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # Precompute factorials up to n!
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
        
        # Convert k to zero-based index
        k -= 1
        
        # Create list of numbers from 1 to n
        numbers = list(range(1, n + 1))
        
        # Call the recursive helper function
        return self.helper(numbers, k, fact)
    
    def helper(self, remaining, k, fact):
        if not remaining:
            return ""
        
        len_remaining = len(remaining)
        current_fact = fact[len_remaining - 1]  # (len_remaining - 1)!
        
        # Determine digit to pick
        index = k // current_fact
        digit = remaining[index]
        
        # Prepare the new list of remaining digits
        new_remaining = remaining[:index] + remaining[index+1:]
        
        # Recurse with the new remaining list and updated k
        return str(digit) + self.helper(new_remaining, k % current_fact, fact)