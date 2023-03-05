from colossalai.amp import AMP_TYPE

BATCH_SIZE = 1
NUM_EPOCHS = 3

fp16 = dict(
    mode=AMP_TYPE.TORCH
)