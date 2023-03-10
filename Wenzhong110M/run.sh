python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr='localhost' \
    --master_port=29500 \
    run.py