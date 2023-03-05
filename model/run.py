import colossalai
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, RandomSampler


hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
# hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'

train_data_path = ''
valid_data_path = ''

params = {
    'max_seq_len': 512,
    'lr': 7e-6,
    'batch_size': 1,
    'lr_scheduler_patience': 3,
    'lr_scheduler_decay': 0.5
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
    # 1.initialize distributed environment
    colossalai.launch_from_torch(config='./config.py')

    # 2.Create training components
    model = GPT2LMHeadModel.from_pretrained(hf_model_path)

    train_dataset = DialogueDataset(data_file=train_data_path, max_seq_length=params['max_seq_len'])
    valid_dataset = DialogueDataset(data_file=valid_data_path, max_seq_length=params['max_seq_len'])
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=params['batch_size'],
                                  collate_fn=train_dataset.batch_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  sampler=RandomSampler(valid_dataset),
                                  batch_size=params['batch_size'],
                                  collate_fn=valid_dataset.batch_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token, reduction='none')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=params['lr_scheduler_decay'], patience=params['lr_scheduler_patience'], verbose=True
    )

    # 3.Get engine object
    engine, train_dataloader, valid_dataloader, _ = colossalai.initialize(model,
                                                                          optimizer,
                                                                          criterion,
                                                                          train_dataloader,
                                                                          valid_dataloader)
    
    # 4.Start training
    logger = get_dist_logger()

    for epoch in range(gpc.config.NUM_EPOCHS):
        # execute a training iteration
        engine.train()
        for img, label in train_dataloader:
            img = img.cuda()
            label = label.cuda()

            # set gradients to zero
            engine.zero_grad()

            # run forward pass
            output = engine(img)

            # compute loss value and run backward pass
            train_loss = engine.criterion(output, label)
            engine.backward(train_loss)

            # update parameters
            engine.step()

        # update learning rate
        lr_scheduler.step()

        # execute a testing iteration
        engine.eval()
        correct = 0
        total = 0
        for img, label in valid_dataloader:
            img = img.cuda()
            label = label.cuda()

            # run prediction without back-propagation
            with torch.no_grad():
                output = engine(img)
                test_loss = engine.criterion(output, label)

            # compute the number of correct prediction
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])