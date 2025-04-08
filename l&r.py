max1=0
l=int(input("Enter l:"))
r=int(input("Enter r:"))
if l>r:
    l,r=r,l
for i in range(l,r+1):
    for j in range(i,r+1):
        v=(i^j)
        
        if max1<v:
            max1=v
            l1=i
            l2=j
print(f"{l1}^{l2}:",max1)