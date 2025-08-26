import math
from sentence_transformers import SentenceTransformer
import heapq
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# the key for qwen-turbo, for translation 
api_key = os.getenv('qwen_api')

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



def translate(text, tolerance=3):
  t = 0 # tolerance
  result = 'error'
  while t<tolerance:
    try:
      client = OpenAI(
          api_key=api_key,
          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
      )

      completion = client.chat.completions.create(
          model="qwen-turbo",
          messages=[
              {"role": "system", "content": 'you are translating category names in the context of online retailer'},
              {"role": "user", "content": f'translate it into english, only output your translation:\n {text}'},
          ],
          stream=False,
          extra_body={"enable_thinking": False}
      )
      response_json = completion.model_dump(mode='json')
      result = response_json['choices'][0]['message']['content']
      break
    except:
      print(f'fail {t+1} ')
      t+=1
      continue

  return result



def embedding_one(text_list):
  return model.encode(text_list, batch_size=12)
