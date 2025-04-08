def all_occurrences(s, pattern):
    start = 0
    while True:
        start = s.find(pattern, start)
        if start == -1: return
        yield start
        start += len(pattern) 

s = input("ENTER TEXT:")
pattern = input("ENTER PATTERN:")
print(list(all_occurrences(s, pattern)))  
