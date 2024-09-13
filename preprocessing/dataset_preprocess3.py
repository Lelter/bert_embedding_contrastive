import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vllm import LLM, SamplingParams
#大模型生成数据测试
pandas.set_option('display.max_columns', None)

os.chdir('/data/yyt/bert_embedding_contrastive/bert_embedding_contrastive')


def load_data():
    data_path = './data/ml-1m/remap_data.csv'

    # read data
    df = pandas.read_csv(data_path)

    # print(df)
    # rating=3删除
    df['label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
    df = df[df['rating'] != 3]
    # 按照时间戳排序
    df = df.sort_values(by=['timestamp'])

    # 统计特征数量
    print(df.head())
    total_features_dims = 0
    for column in df.columns:
        total_features_dims += len(df[column].unique())
    print(total_features_dims)
    # json.dump(total_features_dims, open('./data/ml-1m/total_features_dims.json', 'w'))
    df.to_csv('./data/ml-1m/struct_data.csv', index=False)
    df = df.drop(columns=['timestamp', 'rating'])

    result = (
            'This is a user,'
            'gender is ' + df['gender'] + ',' +
            'age is ' + df['age'].astype(str) + ',' +
            'occupation is ' + df['occupation'].astype(str) + ',' +
            'zip is ' + df['zip'].astype(str) + '.' + 'This is a movie,'
                                                      'title is ' + df['title'] + ',' + 'genres is ' + df[
                'genres'] + '.'
    )
    df['result'] = result
    df['result'].to_csv('./data/ml-1m/text.txt', index=False)

    print(df)


def prompt_preference():
    # 生成prompt:give user_id:{},movie_id:{},outputs <pref> to indicate the preference of the user to the movie.
    data_path = './data/ml-1m/struct_data.csv'
    df = pandas.read_csv(data_path)
    result = ("give user_id:" + df['user_id'].astype(str) + ",movie_id:" + df['movie_id'].astype(
        str) + ",outputs <pref> to indicate the preference of the user to the movie.")
    df['result'] = result
    print(df['result'].head())
    df['result'].to_csv('./data/ml-1m/prompt.txt', index=False)

    pass


def embedding_pseudo_labels():
    # 读取数据
    data_path = './data/ml-1m/struct_data.csv'
    df = pd.read_csv(data_path)

    # 为每个特征生成伪标签，格式化列
    for column in df.columns:
        df[column] = column + ' is ' + df[column].astype(str)

    # 删去user_id、movie_id、rating、timestamp和label列
    df = df.drop(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'label'])

    # 转为一列，每行为一个特征
    df = df.stack().reset_index(drop=True)

    # 定义伪标签映射
    pseudo_labels = {'title': 0, 'genres': 1, 'gender': 2, 'age': 3, 'occupation': 4, 'zip': 5}

    # 生成伪标签
    labels = []
    for item in df:
        for key in pseudo_labels.keys():
            if key in item:
                labels.append(pseudo_labels[key])
                break

    # 创建包含特征和对应伪标签的数据框
    pseudo_label_df = pd.DataFrame({'feature': df, 'pseudo_label': labels})

    print(pseudo_label_df.info())
    pseudo_label_df.to_csv('./data/ml-1m/pseudo_label_data.csv', index=False)
    return pseudo_label_df


def tokenize_batch(batch, tokenizer):
    # prompt = (f'Next, when I send you a sentence, you should replace subject with completely different synonyms but do not replace the Object words. for '
    #           f'instance, change gender to sex and zip to post and title to name and genre to catogory. You cannot simply remove, add, '
    #           f'or modify just a couple of characters. You may insert some meaningless yet non-influential words in '
    #           f'sentences where replacements are not feasible, and you can also alter the word order as long as it does '
    #           f'not affect the meaning. You must strictly adhere to this guideline.')
    # prompt = (f'Generate a sentence that follows the format "aa is bb", replacing "aa" with a synonym while keeping "bb" unchanged. Rewrite the sentence structure, adding words if needed, but strictly adhering to the given format.')
    # prompt = (f"Rephrase the sentence by changing 'aa' to synonyms and altering the structure while keeping 'bb' unchanged, using the format: [synonym of aa] is [bb].")
    # prompt = (f"Transform the sentence 'aa is bb' by substituting 'aa' with various synonyms, altering the sentence structure while maintaining 'bb' in sentence unchanged. You may include additional words.Only output one sentence.")
    prompt = (f"""Instructions
Rewrite the following subject-verb-object sentence, ensuring that the subject is different from the original sentence. You can change the sentence structure and expand it.

Example
The cat chased the mouse.

Output
Startled by the sudden movement, the mouse was chased by a sleek and agile feline through the narrow alleyway.""")
    # prompt = (f'Next, when I send you a sentence, you should expand the subject with a synonym without changing its meaning;for '
    #           f'example,word "zip","gender","title","genre","occupation","age" must be expand.'
    #           f' Using abbreviations is permitted.Respond only in English. You must strictly adhere to this guideline.')
    # prompt="please expand the following sentence by replacing the words 'title,' 'zip,', 'genres,','age','occupation','gender' with their synonyms. Respond only in English."
    return [
        tokenizer.apply_chat_template([{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
        for text in batch]
    # return [
    #     tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    #     for text in batch]


def data_enhancement():
    text_csv = './data/ml-1m/pseudo_label_data.csv'
    df = pd.read_csv(text_csv)
    # 统计tokens数

    df = df[:400]
    max_model_len, tp_size = 8192, 1
    # model_name = "/data/llm/LLM/Meta-Llama-3-8B-Instruct/"
    # model_name = "/data/llm/gemma-2-9b-it/"#do no support
    model_name = "./pretrained_models/THUDM/glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=True)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,

        # dtype="bfloat16"
    )

    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50,min_p=0.1,
                                     stop_token_ids=stop_token_ids,)
    # 先对所有文本进行tokenizer处理
    # queries = [(f"Please rewrite the content below by replacing the first word with a synonym and using different phrases and structures to reformulate the text:\n{text}")
    #            for text in df['feature']]
    # queries = [(f'Please rewrite the following sentence by replacing the subject with a synonym and modifying the '
    #             f'sentence structure, while keeping the original meaning intact.The sentence may be incomplete. For example,'
    #             f'"gender is m" should be rewritten as "sex is male." and "title" should be rewritten as '
    #             f'"name","zip" should be rewritten as "postal code" Respond only in English.\n\n'
    #             f'{text}') for text in df['feature']]
    # queries = [(f'Next, when I send you a sentence, you should replace as many words as possible with synonyms; for '
    #             f'instance, change gender to sex and zip to post and title to name. You cannot simply remove, add, or modify just a '
    #             f'couple of characters. You may insert some meaningless yet non-influential words in sentences where '
    #             f'replacements are not feasible, and you can also alter the word order as long as it does not affect '
    #             f'the meaning. Using abbreviations is permitted. You must strictly adhere to this guideline.\n\n'
    #             f'{text}') for text in df['feature']]
    queries = [f'{text}' for text in df['feature']]

    # queries = [(f'Please rewrite the following sentence by replacing the words "title," "zip," "genres," "gender," and "occupation"'
    #            f' with their synonyms. Respond only in English.\n\n{text}') for text in df['feature']]

    # 批处理tokenizer
    tokenizer_batch_size = 2048
    tokenizer_batches = [queries[i:i + tokenizer_batch_size] for i in range(0, len(queries), tokenizer_batch_size)]

    # 并行处理tokenizer
    inputs = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # 根据你的CPU核心数调整
        futures = [executor.submit(tokenize_batch, batch, tokenizer) for batch in tokenizer_batches]
        for future in tqdm(futures, total=len(tokenizer_batches), desc="Tokenizing"):
            inputs.extend(future.result())
    print("start output")
    outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
    print(len(outputs))
    rephrased_texts = [output.outputs[0].text.strip() for output in outputs]
    print(len(rephrased_texts))
    df['rephrase'] = rephrased_texts

    df.to_csv('./data/ml-1m/rephrase_data3.csv', index=False)


# prompt_preference()
# load_data()
# embedding_pseudo_labels()
data_enhancement()
