class Solution:
    def __init__(self):
        self.mp = {}

    def isScramble(self, s1: str, s2: str) -> bool:
        n = len(s1)
        
        if s1 == s2:
            return True
        if n == 1:
            return False

        key = s1 + " " + s2

        if key in self.mp:
            return self.mp[key]

        for i in range(1, n):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                self.mp[key] = True
                return True

            if self.isScramble(s1[:i], s2[n - i:]) and self.isScramble(s1[i:], s2[:n - i]):
                self.mp[key] = True
                return True

        self.mp[key] = False
        return False