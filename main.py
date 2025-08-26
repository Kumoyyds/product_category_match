import pandas as pd
import concurrent.futures
from tqdm import tqdm
import numpy as np
import joblib
import os
from pathlib import Path
print('loading match funcs...')
from tools import match_func as match
from tools import trans 
print('succeed\n')
import yaml




BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)





with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
output_path = 'output/' + config['output_path']


print('reading the input files...')
file_path = 'input_data/' + config['input_file']
cate_path = 'input_data/' + config['target_file']
file = pd.read_excel(file_path)
cate = pd.read_excel(cate_path)
print('succeed\n')


# read the parameters
file_cols = config['file_cols']
cate_cols = config['cate_cols']
cate_main_num = config['cate_main_num']
consider_sku_name = config['consider_sku_name']
sku_cate_num = config['sku_cate_num']


trans_file = config['trans_file']
trans_file_path = 'cache/' + config['trans_file_path']

trans_cate = config['trans_cate']
trans_cate_path = 'cache/' + config['trans_cate_path']

# do the cleaning
for col in file_cols:
  file[col] = file[col].str.strip()

for col in cate_cols:
  cate[col] = cate[col].str.strip()


# prepare the sentence before embedding
sentence = set()
for col in file_cols:
  sentence = sentence | set(file[col])

sentence_cate = set()
for col in cate_cols:
  sentence_cate = sentence_cate | set(cate[col])

def remove_na(it):
  it = list(it)
  i = 0
  while i < len(it):
    if isinstance(it[i], str):
      i += 1
    else:
      it = it[:i] +  it[i+1:]
  return it

sentence = remove_na(sentence)
sentence_cate = remove_na(sentence_cate) 

##### the final result
print('**begin doing embedding**\n')
print('file embedding')





emb_dic = {}
try:
    cache_t_file = joblib.load(trans_file_path)
except:
    cache_t_file = {}

try:
    cache_t_cate = joblib.load(trans_cate_path)
except:
    cache_t_cate = {}

cache = cache_t_file | cache_t_cate

def translate_with_cache(text, cache = cache):
    if text in cache:
        return cache[text]
    else:
       return trans.translate(text)

   
if trans_file:
  print('Do the translation for sku information\n')
  # then the category
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    sentence_eng = list(tqdm(executor.map(translate_with_cache, sentence), total=len(sentence)))
  print('Do the embedding on english sentence, pls be patient\n')
  embedding = match.embedding_one(sentence_eng)
  translated_file = {i:j for i,j in zip(sentence,sentence_eng)}
  print('save the translation file')
  joblib.dump(translated_file, trans_file_path, compress=("lzma", 3))   # 保存
else:
  embedding = match.embedding_one(sentence)

# load into the cache dictionary
for i in range(len(sentence)):
  emb_dic[sentence[i]] = embedding[i]
print('file embedding succeed\n')


print('target category embedding')
# then the category
if trans_cate:
  print('Do the translation for target cate information')
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    sentence_cate_eng = list(tqdm(executor.map(translate_with_cache, sentence_cate), total=len(sentence_cate)))
  print('Do the embedding on the english cate infor, pls be patient\n')
  embedding_cate = match.embedding_one(sentence_cate_eng)
  translated_cate = {i:j for i,j in zip(sentence_cate, sentence_cate_eng)}
  print('save the translation cate')
  joblib.dump(translated_cate, trans_cate_path, compress=("lzma", 3))   # 保存
else:
  embedding_cate = match.embedding_one(sentence_cate)

# load into the cache dictionary
for i in range(len(sentence_cate)):
  emb_dic[sentence_cate[i]] = embedding_cate[i]
print('target category embedding succeed\n')

print('**all embeddings succeed**\n')


del sentence
del sentence_cate
del embedding
del embedding_cate


### begin the matching 
flex = config['flex']
############################################
if not flex:
    print('normal matching')
    ws = [match.weight(i+1, cate_main_num) for i in range(cate_main_num)]
    match_cate = cate[cate_cols[:cate_main_num]].drop_duplicates()
    match_cate = match_cate.dropna()
    match_cate.reset_index(drop=True, inplace=True)

    # the matrix to be match
    match_matrix = np.vstack(match_cate.apply(lambda row: sum([ws[i]*emb_dic[row[i]] for i in range(cate_main_num)]), axis=1))

    if consider_sku_name:
        match_file_col = file_cols[:sku_cate_num] + [file_cols[-1]]
    else:
        match_file_col = file_cols[:sku_cate_num] + []

    sku_cate_num_f = len(match_file_col)

    # match using this
    ws = [match.weight(i+1, sku_cate_num_f) for i in range(sku_cate_num_f)]


# begin the matching
## this is for toe to toe matching
    def final_match(df):
        add_result = {}
        for i in range(cate_main_num):
            add_result[f'match_cate_{i+1}'] = []
        add_result['sim'] = []

        df.reset_index(drop=True, inplace=True)
        for i in range(df.shape[0]):
            row = list(df.loc[i, match_file_col])
            try:
                v = sum([ws[i]*emb_dic[row[i]] for i in range(sku_cate_num_f)])
                sim = match_matrix @ v
                top_id = match.top_k_indices(sim, 1)[0]

                # match row
                match_row = list(match_cate.loc[top_id])
                for j in range(cate_main_num):
                    add_result[f'match_cate_{j+1}'].append(match_row[j])
                add_result['sim'].append(sim[top_id])
            except:
                for j in range(cate_main_num):
                    add_result[f'match_cate_{j+1}'].append('error')
                add_result['sim'].append('error')

        add_result = pd.DataFrame(add_result)
        df = pd.concat([df, add_result], axis=1)
        return df


    n_split = match.get_it(file.shape[0])
    dfs = np.array_split(file, n_split)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        dfs = list(tqdm(executor.map(final_match, dfs), total=n_split))
        print("matching is finished")

    final = pd.concat(dfs, axis=0)
    final.reset_index(drop=True, inplace=True)

    if trans_cate:
        match_translation = []
        for i in range(final.shape[0]):
            sub = []
            for j in range(cate_main_num):
                sub.append(translated_cate[final.loc[i, f"match_cate_{j+1}"]])
            match_translation.append(" ||| ".join(sub))

        final['match_cate_translation'] = match_translation

else: 
    print('flexible matching')
    mid_result = file.loc[:, file_cols]
    mid_result = mid_result.drop_duplicates()
    # mid_result.rename(columns={i:f"joybuy_{i}" for i in joybuy_cols}, inplace=True)
    mid_result.reset_index(drop=True, inplace=True)


    ws = {i:match.weight(i+1, sku_cate_num) for i in range(sku_cate_num)}
    vs = mid_result.apply(lambda row: sum([ws[i]*emb_dic[row.iloc[i]] for i in range(sku_cate_num)]), axis=1)

    # the list of target categories
    tar_cate = []
    # the matrix of embeddings for target categories
    matrix_result = []
    for i in range(cate_main_num):
        ws = {j:match.weight(j+1, i+1) for j in range(i+1)}
        sub = cate[cate_cols].iloc[:,:i+1].drop_duplicates().dropna()
        tar_cate.extend(sub.apply(lambda row: " ||| ".join(row), axis=1))

        matrix_result.extend(sub.apply(lambda row: sum([ws[j]*emb_dic[row.iloc[j]] for j in range(i+1)]), axis=1))
    matrix_result = np.vstack(matrix_result)

    top_n = config['top_n']
    add_result = {}
    for i in range(top_n):
        add_result[f'cate_match{i+1}_cate'] = []
        add_result[f'cate_match{i+1}_level'] = []
        add_result[f'cate_match{i+1}_sim'] = []



    for i in tqdm(range(mid_result.shape[0])):
        v = vs[i]
        sims = matrix_result @ v
        top_k = match.top_k_indices(sims, top_n)

        for i in range(top_n):
            add_result[f'cate_match{i+1}_cate'].append(tar_cate[top_k[i]])
            add_result[f'cate_match{i+1}_level'].append(tar_cate[top_k[i]].count('|||') + 1)
            add_result[f'cate_match{i+1}_sim'].append(sims[top_k[i]])
    add_result = pd.DataFrame(add_result)
    final = pd.concat([mid_result, add_result], axis=1)

final.reset_index(drop=True, inplace=True)
final.to_excel(output_path, index=False)
print('\n***task is finished, pleae find the result in the ouput folder***\n')