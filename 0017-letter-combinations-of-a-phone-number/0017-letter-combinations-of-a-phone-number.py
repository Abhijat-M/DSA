class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        if not digits:
            return res

        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }

        def helper(pro, unpro):
            if not unpro:
                res.append(pro)
                return
            for char in phone_map[unpro[0]]:
                helper(pro + char, unpro[1:])

        helper("", digits)
        return res
            