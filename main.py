import pandas as pd
import math
import concurrent.futures
from tqdm import tqdm
import numpy as np
import joblib, pickle
import copy
import os

import os
import json
from pathlib import Path

from tools import match_func as match

import yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

file_path = 'input_data/' + config[]
file = pd.read_excel(root + '删除类目存在有效商品统计20250813.xlsx', sheet_name='待删除类目下存在的商品')
cate = pd.read_excel(root + '删除类目存在有效商品统计20250813.xlsx', sheet_name='新类目清单')