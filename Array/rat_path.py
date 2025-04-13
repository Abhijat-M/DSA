"""There is a rat stuck in a maze which wants to get out. The issue over here is when the rat reaches at any cell at a maze it has to take a decision.
Either it can go forward or downward but it cannot ever go backward or upward.
Help the mouse exit the maze by finding all the possible paths from the start to the end. The rat will always start at first cell
and end at the last cell of the maze.
The rat can only move in two directions: down and right. The rat cannot move up or left.
Use recursion to solve the problem."""

def find_paths(maze):
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
    rows = len(maze)
    cols = len(maze[0])
    all_paths = []

    def find_path_recursive(row, col, current_path):
        # Add the current cell to the path
        current_path = current_path + [(row, col)]

        # Base case: Reached the end of the maze
        if row == rows - 1 and col == cols - 1:
            all_paths.append(current_path)
            return

        # Move down
        if row + 1 < rows:
            find_path_recursive(row + 1, col, current_path.copy())

        # Move right
        if col + 1 < cols:
            find_path_recursive(row, col + 1, current_path.copy())

    # Start the recursion from the top-left corner (0, 0)
    find_path_recursive(0, 0, [])
    return all_paths



rows = int(input("Enter the number of rows in the maze: "))
cols = int(input("Enter the number of columns in the maze: "))
maze = [[0] * cols for _ in range(rows)] 

paths = find_paths(maze)

if paths:
    print("All possible paths from start to end:")
    for path in paths:
        print(path)

    print(f"\n Total number of paths: {len(paths)}")
else:
    print("No path found.")
