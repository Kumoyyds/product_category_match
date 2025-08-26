import math
from sentence_transformers import SentenceTransformer
import heapq
import os

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# embeddings = model.encode(sentences)
def top_k_indices(arr, k):
    if k <= 0:
        return []

    # Initialize a min-heap with the first k elements (value, index)
    min_heap = [ (val, idx) for idx, val in enumerate(arr[:k]) ]
    heapq.heapify(min_heap)

    # Iterate through the rest of the array
    for idx, val in enumerate(arr[k:], start=k):
        if val > min_heap[0][0]:  # Compare with smallest element in heap
            heapq.heappop(min_heap)
            heapq.heappush(min_heap, (val, idx))

    # Extract indices from the heap and sort them by value descending
    top_k = sorted(min_heap, key=lambda x: -x[0])
    indices = [idx for val, idx in top_k]

    return indices

# the weight of levl k our of p levels
# i = 1, 2, 3 ....
# aj is to adjust the slope of high level growing, the higher the aj the milder the slope
def weight(k, p, aj=2.5):
  total = sum([math.log(aj+i) for i in range(p)])
  return math.log(k+aj-1)/total




def embedding_one(text_list):
  return model.encode(text_list, batch_size=12)


def get_it(num):
  i = 0
  while num // 10**i > 10:
    i += 1
  tens = 10**i
  return num//tens * tens

