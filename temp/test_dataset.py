from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch

tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-uncased')  # Bert的分词器
bertmodel = BertModel.from_pretrained('../pretrained_models/bert-base-uncased', from_tf=True)  # load the TF model for Pytorch
text = " I love <e> ! "
# 对于一个句子，首尾分别加[CLS]和[SEP]。
text = "[CLS] " + text + " [SEP]"
# 然后进行分词
tokenized_text1 = tokenizer.tokenize(text)
print(tokenized_text1)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
# 分词结束后获取BERT模型需要的tensor
segments_ids1 = [1] * len(tokenized_text1)
tokens_tensor1 = torch.tensor([indexed_tokens1])  # 将list转为tensor
segments_tensors1 = torch.tensor([segments_ids1])
# 获取所有词向量的embedding
word_vectors1 = bertmodel(tokens_tensor1, segments_tensors1)[0]
# 获取句子的embedding
sentenc_vector1 = bertmodel(tokens_tensor1, segments_tensors1)[1]
tokenizer.add_special_tokens({'additional_special_tokens': ["<e>"]})
print(tokenizer.additional_special_tokens)  # 查看此类特殊token有哪些
print(tokenizer.additional_special_tokens_ids)  # 查看其id
tokenized_text1 = tokenizer.tokenize(text)
print(tokenized_text1)
