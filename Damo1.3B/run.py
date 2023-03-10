# finetune_dureader.py
from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers
import os


train_data_path = '/mnt/workspace/workgroup/lizhenyu/blender_data/datasets/msdataset/finetune_train.tsv'
valid_data_path = '/mnt/workspace/workgroup/lizhenyu/blender_data/datasets/msdataset/finetune_valid.tsv'

# dataset_dict = MsDataset.load('DuReader_robust-QG')

# train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
#     .map(lambda example: {'src_txt': example['src_txt'] + '\n'})
# valid_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
#     .map(lambda example: {'src_txt': example['src_txt'] + '\n'})

input_kwargs = {'delimiter': '\t'}
train_dataset = MsDataset.load(train_data_path, **input_kwargs)
valid_dataset = MsDataset.load(valid_data_path, **input_kwargs)

max_epochs = 5

tmp_dir = '/mnt/workspace/workgroup/lizhenyu/zh_ckpt/1.3B/'
tensorboard_logdir = os.path.join(tmp_dir, 'log/')

num_warmup_steps = 200

def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {'type': 'AdamW', 'lr': 1e-4}
    cfg.train.dataloader = {
        'batch_size_per_gpu': 4,
        'workers_per_gpu': 1
    }
    cfg.train.hooks = [
        {
            "type": "BestCkptSaverHook",
            "metric_key": "rouge-1",
            "rule": "max"
        },
        {
            "type": "EvaluationHook",
            "interval": 0.2
        },
        {
            "type": "TextLoggerHook",
            "interval": 50
        },
        {
            "type": "IterTimerHook"
        },
        {
            "type": "TensorboardHook",
            "out_dir": tensorboard_logdir,
            "interval": 50
        },
        {
            "type": "MegatronHook"
        }
    ]
    cfg.preprocessor.sequence_length = 512
    cfg.model.checkpoint_model_parallel_size = 1
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_1.3B',
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn,
    use_fp16=True)

trainer = build_trainer(
    name=Trainers.gpt3_trainer, default_args=kwargs)
trainer.train()