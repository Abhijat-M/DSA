def find(arr, k)->bool:
    m = len(arr)
    n = len(arr[0])

    if k < arr[0][0] or k > arr[m-1][n-1]:
        return False

    for i in range(m):
        if k < arr[i][-1]:
            for j in range(n-1):
                if arr[i][j] == k:
                    return True

    return False

def main():
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ]
    print(find(a, 30))

if __name__ == "__main__":
    main()
