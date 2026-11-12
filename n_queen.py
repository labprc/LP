
# N-Queens Problem using Backtracking
# Generate All Possible Solutions

n = int(input("Enter number of queens: "))

board = [[0 for j in range(n)] for i in range(n)]

count = 0

# Check whether position is safe
def is_safe(row, col):

    # Check column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    i = row - 1
    j = col - 1

    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # Check upper-right diagonal
    i = row - 1
    j = col + 1

    while i >= 0 and j < n:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True


# Backtracking function
def solve(row):

    global count

    # All queens are placed
    if row == n:
        count += 1

        print("\nSolution", count)

        for i in range(n):
            for j in range(n):
                print(board[i][j], end=" ")
            print()

        return

    # Try every column
    for col in range(n):

        if is_safe(row, col):

            board[row][col] = 1

            solve(row + 1)

            # Backtrack
            board[row][col] = 0


# Start placing queens
solve(0)

print("\nTotal Solutions:", count)

# Time Complexity: O(N!) worst-case upper bound
# Space Complexity: O(N^2) for board + O(N) recursion stack
