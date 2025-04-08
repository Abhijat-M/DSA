def print_diagonal_matrix(matrix):
    if not matrix or not matrix[0]:
        return

    rows = len(matrix)
    cols = len(matrix[0])
    
    result = []

    for col in range(cols):
        x, y = 0, col
        while x < rows and y >= 0:
            result.append(matrix[x][y])
            x += 1
            y -= 1

    for row in range(1, rows):
        x, y = row, cols - 1
        while x < rows and y >= 0:
            result.append(matrix[x][y])
            x += 1
            y -= 1

    res= ", ".join(map(str, result))

    return res




rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))

arr = [[int(input(f"Enter element at position ({i+1}, {j+1}): ")) for j in range(cols)] for i in range(rows)]

print("Final spiral matrix: \n", print_diagonal_matrix(arr))