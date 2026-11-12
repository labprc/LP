
# 0/1 Knapsack using Dynamic Programming and Branch & Bound

n = int(input("Enter number of items: "))

profit = []
weight = []

# Enter item details
for i in range(n):
    print("\nItem", i + 1)
    p = int(input("Enter profit: "))
    w = int(input("Enter weight: "))
    profit.append(p)
    weight.append(w)

capacity = int(input("\nEnter knapsack capacity: "))


# ---------------------------------
# 1. Dynamic Programming
# ---------------------------------

dp = [[0 for j in range(capacity + 1)]
      for i in range(n + 1)]

for i in range(1, n + 1):
    for w in range(capacity + 1):

        if weight[i - 1] <= w:
            dp[i][w] = max(
                profit[i - 1] + dp[i - 1][w - weight[i - 1]],
                dp[i - 1][w]
            )
        else:
            dp[i][w] = dp[i - 1][w]

print("\nMaximum Profit using Dynamic Programming:",
      dp[n][capacity])


# ---------------------------------
# 2. Branch and Bound
# ---------------------------------

# Sort items according to profit/weight ratio
items = []

for i in range(n):
    items.append((profit[i], weight[i]))

items.sort(key=lambda x: x[0] / x[1], reverse=True)

best_profit = 0

# Calculate upper bound
def bound(level, current_weight, current_profit):

    if current_weight >= capacity:
        return current_profit

    result = current_profit
    total_weight = current_weight
    j = level

    while j < n and total_weight + items[j][1] <= capacity:
        total_weight += items[j][1]
        result += items[j][0]
        j += 1

    if j < n:
        result += (capacity - total_weight) * (
            items[j][0] / items[j][1]
        )

    return result


# Recursive Branch and Bound
def knapsack(level, current_weight, current_profit):

    global best_profit

    if current_weight <= capacity:
        best_profit = max(best_profit, current_profit)

    if level == n:
        return

    if bound(level, current_weight, current_profit) <= best_profit:
        return

    # Include current item
    if current_weight + items[level][1] <= capacity:
        knapsack(
            level + 1,
            current_weight + items[level][1],
            current_profit + items[level][0]
        )

    # Exclude current item
    knapsack(level + 1, current_weight, current_profit)


knapsack(0, 0, 0)

print("Maximum Profit using Branch and Bound:", best_profit)


# Time Complexity:
# Dynamic Programming: O(n * W)
# Branch and Bound: O(2^n) worst case

# Space Complexity:
# Dynamic Programming: O(n * W)
# Branch and Bound: O(n) recursion stack
