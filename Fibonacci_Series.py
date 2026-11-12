
# Fibonacci Series using Recursive and Non-Recursive Methods

# Non-Recursive Function
# Time Complexity: O(n)
# Space Complexity: O(1)

def fibonacci_iterative(n):

    a = 0
    b = 1
    c = 0

    for i in range(n):
        print(a, end=" ")
        c = a + b
        a = b
        b = c


# Recursive Function
# Time Complexity: O(2^n)
# Space Complexity: O(n)

def fibonacci_recursive(n):

    if n == 0:
        return 0

    elif n == 1:
        return 1

    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


# Main Program

n = int(input("Enter number of terms: "))

print("\nFibonacci Series using Non-Recursive Method:")
fibonacci_iterative(n)

print("\nFibonacci Series using Recursive Method:")
for i in range(n):
    print(fibonacci_recursive(i), end=" ")

print()
