from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import torch
import colossalai
from colossalai.utils import load_checkpoint

ckpt_dir = '/mnt/workspace/workgroup/lizhenyu/zh_ckpt/110M-colossalai/colossal_model.pt'
config_path = '/mnt/workspace/workgroup/lizhenyu/zh-Dialogue-exp/Wenzhong3.5B/config.py'
wenzhong_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'

colossalai.launch_from_torch(config=config_path)

model = GPT2LMHeadModel.from_pretrained(wenzhong_path)
load_checkpoint(ckpt_dir, model)

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
                        output_scores=True,
                        num_return_sequences=5)

for idx, sent in enumerate(output):
    print('next sent %d:\n' % idx, tokenizer.decode(sent).split('<|endoftext|>')[0])
    print('*'*40)