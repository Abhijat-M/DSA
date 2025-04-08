def all_occurrence(s, p):
    i = 0
    while i < len(s):
        i = s.find(p, i)
        if i == -1:
            break
        yield i
        i += 1

# def Replace(list(all_occurrence(s,p)), s, p, r):
#     for i in all_occurrence:
#         s = s[:i] + r + s[i+len(p):]
#     return s

s = input("ENTER TEXT:")
p = input("ENTER PATTERN:")
r = input("ENTER REPLACEMENT:")
occur= list(all_occurrence(s, p))
for i in occur:
        s = s[:i] + r + s[i+len(p):]
print(s)
#print(Replace(all_occurrence(s, p), s, p, r))