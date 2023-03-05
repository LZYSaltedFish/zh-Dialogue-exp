import colossalai
from colossalai.logging import get_dist_logger

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset


hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
# hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'

train_data_path = ''
valid_data_path = ''

params = {
    'max_seq_len': 512
}

class DialogueDataset(Dataset):
    def __init__(self,
                 data_file,
                 max_seq_length):
        super().__init__()

        self.data_rows = self.readlines_from_file(data_file)

        self.max_seq_length = max_seq_length

        self.load_tokenizer()
    
    def readlines_from_file(self, data_file):
        print(f'****{data_file}')
        with open(data_file, 'r') as f:
            data_rows = f.readlines()
        return data_rows

    def load_tokenizer(self, pad_token='<pad>', bos_token='<bos>', truncation_side='left'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
        self.tokenizer.add_tokens(pad_token)
        self.tokenizer.add_tokens(bos_token)
        self.tokenizer.pad_token = pad_token
        self.tokenizer.bos_token = bos_token
        self.tokenizer.truncation_side = truncation_side
    
    def __getitem__(self, index):
        row = self.data_rows[index].strip('\n')
        return self.convert_single_row_to_example(row)

    def __len__(self):
        return len(self.data_rows)
    
    def convert_single_row_to_example(self, row):
        sentences = row.split('\t')
        episodes = []
        history = ''
        for turn in range(len(sentences) // 2):
            text = sentences[turn * 2].replace('\\n', '\n') + self.tokenizer.eos_token
            text = history + text
            label = sentences[(turn * 2) + 1].replace('\\n', '\n') + self.tokenizer.eos_token
            history = text + label
            if text == '__null__' or label == '__null__':
                break

            encoded_input = self.tokenizer(history,
                                           padding='max_length',
                                           truncation=True,
                                           max_length=self.max_text_length)
            encoded_input.update({'label_ids': encoded_input['input_ids']})
            episodes.append(encoded_input)
        return episodes

    def batch_fn(self, features):
        refactor_feat = [ep for session in features for ep in session]
        return {k: torch.tensor([dic[k] for dic in refactor_feat]) for k in refactor_feat[0]}

if __name__ == '__main__':
    colossalai.launch_from_torch(config='./config.py')

    model = GPT2LMHeadModel.from_pretrained(hf_model_path)

    train_dataset = DialogueDataset(data_file=train_data_path, max_seq_length=params['max_seq_len'])
    valid_dataset = DialogueDataset(data_file=valid_data_path, max_seq_length=params['max_seq_len'])

    logger = get_dist_logger()