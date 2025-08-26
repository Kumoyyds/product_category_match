from dotenv import load_dotenv
load_dotenv()
import os
api = os.getenv('qwen_api')
print(api)