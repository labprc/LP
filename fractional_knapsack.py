
# Fractional Knapsack using Greedy Method

n = int(input("Enter number of items: "))

profit = []
weight = []

# Enter item details one by one
for i in range(n):
    print("\nItem", i + 1)
    p = int(input("Enter profit: "))
    w = int(input("Enter weight: "))

    profit.append(p)
    weight.append(w)

capacity = int(input("\nEnter knapsack capacity: "))

# Calculate profit/weight ratio
ratio = []

for i in range(n):
    ratio.append(profit[i] / weight[i])

# Sort according to highest ratio
for i in range(n):
    for j in range(i + 1, n):
        if ratio[i] < ratio[j]:
            ratio[i], ratio[j] = ratio[j], ratio[i]
            profit[i], profit[j] = profit[j], profit[i]
            weight[i], weight[j] = weight[j], weight[i]

total_profit = 0

# Select items using Greedy method
for i in range(n):
    if weight[i] <= capacity:
        total_profit += profit[i]
        capacity -= weight[i]
    else:
        total_profit += ratio[i] * capacity
        break

print("\nMaximum Profit:", total_profit)

# Time Complexity: O(n^2)
# Space Complexity: O(n)
