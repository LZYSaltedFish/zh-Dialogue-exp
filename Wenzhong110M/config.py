from colossalai.amp import AMP_TYPE

BATCH_SIZE = 6
NUM_EPOCHS = 10

fp16 = dict(
    mode=AMP_TYPE.TORCH
)