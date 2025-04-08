def kangaroo(x1,x2,v1,v2):
    if v1!=v2 and (v1*int(abs((x2-x1)/(v1-v2)))+x1) == (v2*int(abs((x2-x1)/(v1-v2)))+x2):
        return True
    else:
        return False 
    
x1=int(input("Enter distance for 1:"))
x2=int(input("Enter distance for 2:"))
v1=int(input("Enter velocity for 1:"))
v2=int(input("Enter velocity for 2:"))
print(kangaroo(x1,x2,v1,v2))