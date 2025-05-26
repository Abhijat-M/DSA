class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        digit_to_letters_map = [
            "0", "0", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
        ]
        result = self._solve1_recursive(digits, digit_to_letters_map)
        return result

    def _solve1_recursive(self, digits: str, mapping: List[str]) -> List[str]:
        if not digits:
            return [""]

        first_digit_char = digits[0]
        digit_value = int(first_digit_char)
        letters_for_first_digit = mapping[digit_value]
        
        remaining_digits = digits[1:]
        combinations_from_rest = self._solve1_recursive(remaining_digits, mapping)
        current_result = []

        for letter in letters_for_first_digit:
            for combo in combinations_from_rest:
                current_result.append(letter + combo)
        
        return current_result

           