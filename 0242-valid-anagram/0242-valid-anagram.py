class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s=[i for i in s]
        t=[i for i in t]
        
        s=Counter(s)
        t=Counter(t)
        return s==t
        