from dotenv import load_dotenv
import os
from openai import OpenAI
import joblib
from pathlib import Path
# the key for qwen-turbo, for translation 
load_dotenv()
api_key = os.getenv('qwen_api')


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
          model="qwen-turbo-latest",
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
