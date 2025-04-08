def sort(arr):
    n = len(arr)
    one=0
    two=n
    new_arr = []
    k=n-1

    for one, two in zip(range(0,n), range(n,0,-1)):

        if arr[one] > arr[two]:
            new_arr[k]=arr[one]
            k-=1
            one+=1
        elif arr[one] < arr[two]:
            new_arr[k]=arr[two]
            k-=1
            two-=1
        elif arr[one] == arr[two]:
            exit(1)
        
    return new_arr


arr=eval(input("Enter the array: "))
arr= [i**2 for i in arr]
print(sort(arr))
