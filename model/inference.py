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
encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

gen_config = GenerationConfig(min_new_tokens=10, max_new_tokens=80, num_beams=10)
output = model.generate(**encoded_input,
                        generation_config=gen_config,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        return_dict_in_generation=True,
                        output_scores=True,
                        num_return_sequences=5)

for idx, sent in enumerate(output.sequences):
    print('next sent %d:\n' % idx, tokenizer.decode(sent).split('<|endoftext|>')[0])
    print('*'*40)