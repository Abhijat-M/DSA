"""There is a rat stuck in a maze which wants to get out. The issue over here is when the rat reaches at any cell at a maze it has to take a decision.
Either it can go forward or downward but it cannot ever go backward or upward.
Help the mouse exit the maze by finding all the possible paths from the start to the end. The rat will always start at first cell
and end at the last cell of the maze.
The rat can only move in two directions: down and right. The rat cannot move up or left.
Use DP to solve the problem."""

def find_paths(maze: list[list[int]]) -> list[list[str]]:
    """
    Finds all possible paths for a rat to exit a maze.

    Args:
        maze: A 2D list representing the maze. The dimensions of the maze
              determine the start (top-left) and end (bottom-right) points.

    Returns:
        A list of lists, where each inner list represents a path from the
        start to the end of the maze. Each path is a sequence of tuples,
        where each tuple represents a cell (row, column).
    """
    m = len(maze)
    n = len(maze[0])
    all_paths = []

    def is_valid(x, y):
        return 0 <= x < m and 0 <= y < n and maze[x][y] == 1

    def backtrack(x, y, path):
        if x == m - 1 and y == n - 1:
            all_paths.append(path[:])
            return

        # Move Right
        if is_valid(x, y + 1):
            path.append("R")
            backtrack(x, y + 1, path)
            path.pop()

        # Move Down
        if is_valid(x + 1, y):
            path.append("D")
            backtrack(x + 1, y, path)
            path.pop()

    if maze[0][0] == 1:
        backtrack(0, 0, [])

    return all_paths



rows = int(input("Enter the number of rows in the maze: "))
cols = int(input("Enter the number of columns in the maze: "))
maze = [[1] * cols for _ in range(rows)] 

paths = find_paths(maze)
print("All possible paths (Right=R, Down=D):")
for path in paths:
    print("".join(path))

print(f"\nTotal number of paths: {len(paths)}")
