from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import torch

model_path = '/apsarapangu/disk3/xianyu-anaconda/ckpt/Wenzhong-110M/colossal_model.pt'
wenzhong_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(wenzhong_path)

pad_token = '<pad>'
bos_token = '<bos>'
tokenizer.add_tokens(pad_token)
tokenizer.add_tokens(bos_token)
tokenizer.pad_token = pad_token
tokenizer.bos_token = bos_token
tokenizer.truncation_size = 'left'

text = "你今天看了那场球赛了吗"
encoded_input = tokenizer(text)