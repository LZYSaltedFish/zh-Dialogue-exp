import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import math
from tqdm import tqdm
import time

import colossalai
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import save_checkpoint, load_checkpoint

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
# hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'

config_path = '/apsarapangu/disk3/xianyu-anaconda/zh-Dialogue-exp/model/config.py'

train_data_path = '/apsarapangu/disk3/xianyu-anaconda/zh-Dialogue-exp/dataset/plain_chitchat/pretrain_train.tsv'
valid_data_path = '/apsarapangu/disk3/xianyu-anaconda/zh-Dialogue-exp/dataset/plain_chitchat/pretrain_valid.tsv'

ckpt_dir = '/apsarapangu/disk3/xianyu-anaconda/ckpt/Wenzhong-110M/'

huggingface_cache_path = '/apsarapangu/disk3/xianyu-anaconda/huggingface_cache'

params = {
    'max_seq_len': 512,
    'lr': 7e-6,
    'lr_scheduler_patience': 3,
    'lr_scheduler_decay': 0.5,
    'save_ckpt_steps': 2000,
    'log_n_steps': 50
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
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path, cache_dir=huggingface_cache_path)
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
                                           max_length=self.max_seq_length)
            encoded_input.update({'label_ids': encoded_input['input_ids']})
            episodes.append(encoded_input)
        return episodes

    def batch_fn(self, features):
        refactor_feat = [ep for session in features for ep in session]
        return {k: torch.tensor([dic[k] for dic in refactor_feat]) for k in refactor_feat[0]}

def compute_token_loss(forward_outputs, label_ids, criterion):
    label_ids = label_ids[:, 1:]
    logits = forward_outputs['logits'][:, :-1]
    logits_view = logits.reshape(-1, logits.size(-1))
    loss = criterion(logits_view, label_ids.reshape(-1))
    loss = loss.sum()
    return loss

def compute_loss(forward_outputs, label_ids, criterion, pad_token_id):
    loss = compute_token_loss(forward_outputs, label_ids, criterion)

    notnull = label_ids.ne(pad_token_id)
    target_tokens = notnull.long().sum()
    loss /= target_tokens
    return loss

def compute_metrics(forward_outputs, label_ids, criterion, pad_token_id):
    label_ids = label_ids[:, 1:]
    logits = forward_outputs['logits'][:, :-1]
    preds = logits.max(dim=-1)[1]
    logits_view = logits.reshape(-1, logits.size(-1))
    loss = criterion(logits_view, label_ids.reshape(-1))
    prob = torch.softmax(logits, dim=-1)
    loss = loss.view(prob.shape[:-1]).sum(dim=1)

    notnull = label_ids.ne(pad_token_id)
    target_tokens = notnull.long().sum(dim=1)
    correct = ((label_ids == preds) * notnull).sum(dim=-1)

    loss = loss.tolist()
    target_tokens = target_tokens.tolist()
    correct = correct.tolist()

    loss_avg = [l/t for l in loss for t in target_tokens]
    total_loss = sum(loss_avg)
    total_ppl = sum([math.exp(i) for i in loss_avg])
    total_acc = sum([c/t for c in correct for t in target_tokens])
    total_em = sum([correct[i]==target_tokens[i] for i in range(len(target_tokens))])
    total_samples = len(target_tokens)
    return total_loss, total_ppl, total_acc, total_em, total_samples

def log_tensorboard(writer,
                    learning_rate,
                    global_step,
                    current_loss=None,
                    eval_scores=None,
                    is_training=True):
    if is_training:
        writer.add_scalar(tag='losses/loss',
                          scalar_value=current_loss,
                          global_step=global_step)
        writer.add_scalar(tag='learning_rate/lr',
                          scalar_value=learning_rate,
                          global_step=global_step)
    else:
        for metric_name, score in eval_scores:
            writer.add_scalar(tag=f'eval/{metric_name}',
                              scalar_value=score,
                              global_step=global_step)


_global_step = 0
best_valid_score = float('inf')

tensorboard = SummaryWriter(
    log_dir=os.path.join(ckpt_dir, 'log'))

last_time = time.time()

if __name__ == '__main__':
    # 1.initialize distributed environment
    colossalai.launch_from_torch(config=config_path)

    # 2.Create training components
    model = GPT2LMHeadModel.from_pretrained(hf_model_path, cache_dir=huggingface_cache_path)

    train_dataset = DialogueDataset(data_file=train_data_path, max_seq_length=params['max_seq_len'])
    valid_dataset = DialogueDataset(data_file=valid_data_path, max_seq_length=params['max_seq_len'])
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=gpc.config.BATCH_SIZE,
                                  collate_fn=train_dataset.batch_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  sampler=RandomSampler(valid_dataset),
                                  batch_size=gpc.config.BATCH_SIZE,
                                  collate_fn=valid_dataset.batch_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id, reduction='none')
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
    total_epochs = gpc.config.NUM_EPOCHS
    total_training_steps = int(math.ceil(len(train_dataloader) * total_epochs))

    for epoch in range(total_epochs):
        # execute a training iteration
        engine.train()
        for _step, batch in enumerate(train_dataloader):
            try:
                batch = {
                    key: val.cuda() if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}
            
            label = batch.pop('label_ids')

            # set gradients to zero
            engine.zero_grad()

            # run forward pass
            output = engine(**batch)


            # compute loss value and run backward pass
            train_loss = compute_loss(output, label, engine.criterion, train_dataset.tokenizer.pad_token_id)
            # train_loss = engine.criterion(output, label)
            engine.backward(train_loss)

            # update parameters
            engine.step()

            _global_step += 1
            cur_lr = optimizer.param_groups[0]['lr']
            log_tensorboard(writer=tensorboard,
                            learning_rate=cur_lr,
                            current_loss=train_loss,
                            global_step=_global_step)
            if (_global_step+1) % params['log_n_steps'] == 0:
                logger.info(f'Epoch [{epoch+1:2}/{total_epochs:2}], step [{_global_step+1}/{total_training_steps}], lr {cur_lr:.6f}, {time.time()-last_time:.2f} s')
                logger.info(f'loss: {train_loss:.4f}')
                last_time = time.time()



            if (_global_step+1) % params['save_ckpt_steps'] == 0:
                # Evaluation
                logger.info(
                    f'========== Evaluation at global step {_global_step+1} =========='
                )
                # execute a testing iteration
                engine.eval()
                total_loss = 0
                total_ppl = 0
                total_acc = 0
                total_em = 0
                total_samples = 0
                for _valid_step, batch in enumerate(tqdm(valid_dataloader)):
                    try:
                        batch = {
                            key: val.cuda() if isinstance(val, torch.Tensor) else val
                            for key, val in batch.items()
                        }
                    except RuntimeError:
                        batch = {key: val for key, val in batch.items()}


                    # run prediction without back-propagation
                    with torch.no_grad():
                        label = batch.pop('label_ids')
                        output = engine(**batch)
                        loss, ppl, acc, em, samples = compute_metrics(output, label, engine.criterion, valid_dataset.tokenizer.pad_token_id)
                        # valid_loss = engine.criterion(output, label)

                    # compute the metrics
                    total_loss += loss
                    total_ppl += ppl
                    total_acc += acc
                    total_em += em
                    total_samples += samples
                
                # cross entropy loss
                total_loss /= total_samples
                # perplexity
                total_ppl /= total_samples
                # token-wise accuracy
                total_acc /= total_samples
                # utterance-wise exact match
                total_em /= total_samples

                logger.info(
                    f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {total_loss:.5}, ppl: {total_ppl:.5}, acc: {total_acc:.5}, lr: {optimizer.param_groups[0]['lr']:.5g}", ranks=[0])
                
                
                # Save model
                if total_loss < best_valid_score:
                    saving_path = os.path.join(ckpt_dir, 'colossal_model.pt')
                    logger.info(
                        f'Saving best model to {saving_path}...'
                    )
                    save_checkpoint(saving_path, epoch, model)
                    best_valid_score = total_loss
                
                metrics = [('loss', total_loss), ('ppl', total_ppl), ('acc', total_acc), ('em', total_em)]
                log_tensorboard(writer=tensorboard,
                                learning_rate=optimizer.param_groups[0]['lr'],
                                eval_scores=metrics,
                                global_step=_global_step,
                                is_training=False)
                
                # update learning rate
                lr_scheduler.step(total_loss)

        