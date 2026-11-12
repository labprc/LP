
import heapq

# Node class
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# Generate Huffman Codes
def generate_codes(root, code, codes):

    if root is None:
        return

    if root.char is not None:
        codes[root.char] = code
        return

    generate_codes(root.left, code + "0", codes)
    generate_codes(root.right, code + "1", codes)


# Huffman Encoding Function
# Time Complexity: O(n log n)
# Space Complexity: O(n)

def huffman_encoding(chars, freq):

    heap = []

    for i in range(len(chars)):
        node = Node(chars[i], freq[i])
        heapq.heappush(heap, node)

    while len(heap) > 1:

        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        new_node = Node(None, left.freq + right.freq)

        new_node.left = left
        new_node.right = right

        heapq.heappush(heap, new_node)

    root = heapq.heappop(heap)

    codes = {}
    generate_codes(root, "", codes)

    return codes


# Main Program

n = int(input("Enter number of characters: "))

chars = []
freq = []

for i in range(n):
    ch = input("Enter character: ")
    f = int(input("Enter frequency: "))

    chars.append(ch)
    freq.append(f)

codes = huffman_encoding(chars, freq)

print("\nHuffman Codes:")

for ch in codes:
    print(ch, ":", codes[ch])
