import colossalai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import GeminiDDP
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.zero import ZeroOptimizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
from colossalai.amp import AMP_TYPE

hf_model_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
dataset_path = '/mnt/workspace/workgroup/lizhenyu/blender_data/datasets/finetune_train.tsv'


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
            if '__null__' in text or '__null__' in label:
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

myconfig=dict(fp16=dict(mode=AMP_TYPE.TORCH))
colossalai.launch_from_torch(config=myconfig)
# args = colossalai.get_default_parser().parse_args()
# colossalai.launch(config=myconfig,
#                   rank=args.rank,
#                   world_size=args.world_size,
#                   host=args.host,
#                   port=args.port)

with ColoInitContext(device='cpu'):
    model = GPT2LMHeadModel.from_pretrained(hf_model_path)

PLACEMENT_POLICY = 'cpu'
model = GeminiDDP(model,
                  device=get_current_device(),
                  placement_policy=PLACEMENT_POLICY,
                  pin_memory=True)
LR = 5e-5
optimizer = HybridAdam(model.parameters(), lr=LR)
optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**14)
train_dataset = DialogueDataset(data_file=dataset_path, max_seq_length=512)
train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=1,
                              collate_fn=train_dataset.batch_fn)
criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id, reduction='none')
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.5, patience=3, verbose=True
)

model.train()
for epoch in range(5):
    for step, batch in enumerate(train_dataloader):
        try:
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
        except RuntimeError:
            batch = {key: val for key, val in batch.items()}
        
        label = batch.pop('label_ids')
        optimizer.zero_grad()
        output = model(**batch)
        train_loss = compute_loss(output, label, criterion, train_dataset.tokenizer.pad_token_id)
        optimizer.backward(train_loss)
        optimizer.step()
        lr_scheduler.step(train_loss)