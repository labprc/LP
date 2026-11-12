
# Quick Sort using Deterministic and Randomized Methods

import random
import time

# ---------------------------------
# 1. Deterministic Quick Sort
# ---------------------------------

def partition(arr, low, high):

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):

        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1


def deterministic_quick_sort(arr, low, high):

    if low < high:

        pi = partition(arr, low, high)

        deterministic_quick_sort(arr, low, pi - 1)
        deterministic_quick_sort(arr, pi + 1, high)


# ---------------------------------
# 2. Randomized Quick Sort
# ---------------------------------

def randomized_partition(arr, low, high):

    random_index = random.randint(low, high)

    arr[random_index], arr[high] = arr[high], arr[random_index]

    return partition(arr, low, high)


def randomized_quick_sort(arr, low, high):

    if low < high:

        pi = randomized_partition(arr, low, high)

        randomized_quick_sort(arr, low, pi - 1)
        randomized_quick_sort(arr, pi + 1, high)


# ---------------------------------
# Main Program
# ---------------------------------

n = int(input("Enter number of elements: "))

arr = []

for i in range(n):
    value = int(input("Enter element: "))
    arr.append(value)

# Copy original array
arr1 = arr.copy()
arr2 = arr.copy()

# Deterministic Quick Sort
start = time.time()

deterministic_quick_sort(arr1, 0, n - 1)

end = time.time()

time1 = end - start

print("\nDeterministic Quick Sort:")
print("Sorted Array:", arr1)
print("Execution Time:", time1, "seconds")


# Randomized Quick Sort
start = time.time()

randomized_quick_sort(arr2, 0, n - 1)

end = time.time()

time2 = end - start

print("\nRandomized Quick Sort:")
print("Sorted Array:", arr2)
print("Execution Time:", time2, "seconds")


# Time Complexity:
# Deterministic: Best/Average O(n log n), Worst O(n^2)
# Randomized: Expected O(n log n), Worst O(n^2)

# Space Complexity:
# Both: O(log n) average recursion stack
# Worst case: O(n) recursion stack
