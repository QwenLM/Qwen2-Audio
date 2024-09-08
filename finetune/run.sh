export GPUS_PER_NODE=8
export NCCL_IB_QPS_PER_CONNECTION=8
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0

# Only test deepspeed_z1.yaml but it should be the same for other configs
accelerate launch \
    --config_file accelerate_configs/deepspeed_z1.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_machines $WORLD_SIZE \
    --num_processes $(($WORLD_SIZE * $GPUS_PER_NODE)) \
    run.py \
    --model_name_or_path  Qwen/Qwen2-Audio-7B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --max_train_steps 20000 \
    --trust_remote_code \
    --save_interval 5 \
    --gradient_checkpointing \
    --lora \
    $@