python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_addr='localhost' \
    --master_port=29500 \
    example.py