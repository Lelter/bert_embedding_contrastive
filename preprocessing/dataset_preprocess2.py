import datetime
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)
os.chdir('/data/yyt/bert_embedding_contrastive/bert_embedding_contrastive')
text_path = './data/ml-1m/rephrase_data2.csv'
columns = ['title', 'genres', 'gender', 'age', 'occupation', 'zip', 'pseudo_label']

# 读取数据
data = pd.read_csv(text_path)

def extract_info(row):
    feature = row['feature']
    result = {
        'title': '', 'genres': '', 'gender': '', 'age': '', 'occupation': '', 'zip': '',
        'pseudo_label': row['pseudo_label']
    }

    if 'title' in feature:
        result['title'] = row['rephrase']
    elif 'genres' in feature:
        result['genres'] = row['rephrase']
    elif 'gender' in feature:
        result['gender'] = row['rephrase']
    elif 'age' in feature:
        result['age'] = row['rephrase']
    elif 'occupation' in feature:
        result['occupation'] = row['rephrase']
    elif 'zip' in feature:
        result['zip'] = row['rephrase']

    return pd.Series(result)

def apply_extract_info(df_chunk):
    return df_chunk.apply(extract_info, axis=1)

# 并行处理函数
def parallelize_dataframe(df, func):
    n_cores = cpu_count()  # 使用所有的 CPU 核心数
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

# 应用转换函数并进行并行处理
start_time = datetime.datetime.now()
df_transformed_parallel = parallelize_dataframe(data, apply_extract_info)

# 每6行合并
def combine_rows(df, n=6):
    return df.groupby(df.index // n).agg(lambda x: ', '.join(x.replace('', np.nan).dropna().astype(str))).reset_index(drop=True)

# 合并结果
df_combined = combine_rows(df_transformed_parallel)
end_time = datetime.datetime.now()
print(end_time - start_time)
print(df_combined)

# 保存结果
df_combined.to_csv('./data/ml-1m/rephrase_data2_combine.csv', index=False)