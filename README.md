# product_category_match
## basic info
This is used for updating or matching categories in the context of online retail

contributor: Yuding Duan

the accuary is not good enough for sku name matching (as the target), only do it to match category  
cate->cate √  
sku_name->cate √  
any->sku_name ×  

**supports**:
1. it's mainly for english content matching, but multilingual matching is avaialble as well (needs qwen api tho, apply for it [here](https://bailian.console.aliyun.com/?utm_content=se_1021228139&gclid=EAIaIQobChMIx4_4gpWojwMVsV1HAR3UuDkcEAAYASAAEgLdxPD_BwE#/home)) 
2. it *supports normal matching* (with target category level fixed) and *flexible matching* (not above given thredshold)

## Preparation
1. clone the repo  
`git clone https://github.com/Kumoyyds/product_category_match.git`

2. go to the dir  
`cd product_category_matching`  

3. create a new virtual env (recommended)  
conda, py, or python, whichever you like.  

4. install the packages  
`pip install -r requirements.txt`  

## usage  
(optional) if you need multilingual capability:  
`cp env.sample .env` and then fill your **api_key**  

1. Save your input files in the folder **input_data/** (1. file, the original file   2. target file which to be matched, like newly updated category list)  

2. Go to **config.yaml** to adjust the parameters based on your needs.   

3. run `python main.py` in your terminal.  

4. find your output in the folder **output/**
