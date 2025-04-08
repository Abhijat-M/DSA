class Matrix:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # Traverse from left to right
            for c in range(left, right + 1):
                result.append(matrix[top][c])
            top += 1
            
            # Traverse from top to bottom
            for r in range(top, bottom + 1):
                result.append(matrix[r][right])
            right -= 1
            
            if top <= bottom:
                # Traverse from right to left
                for c in range(right, left - 1, -1):
                    result.append(matrix[bottom][c])
                bottom -= 1
            
            if left <= right:
                # Traverse from bottom to top
                for r in range(bottom, top - 1, -1):
                    result.append(matrix[r][left])
                left += 1
        
        return result
    
obj = Matrix()
rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))

array_2d = [[int(input(f"Enter element at position ({i+1}, {j+1}): ")) for j in range(cols)] for i in range(rows)]

print("Final spiral matrix: \n",obj.spiralOrder(array_2d))
