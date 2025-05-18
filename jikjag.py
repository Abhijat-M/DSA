
def print_matrix(matrix):
  """Prints a 3x3 matrix in a readable format."""
  for row in matrix:
    for element in row:
      print(element, end="\t")
    print()

example_matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

print("3x3 Matrix:")
print_matrix(example_matrix)
