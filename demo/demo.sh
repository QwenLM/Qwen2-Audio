echo $CUDA_VISIBLE_DEVICES
SERVER_PORT=9001
MASTER_ADDR=localhost
MASTER_PORT="3${SERVER_PORT}"
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=1
python -m torch.distributed.launch --use_env \
     --nproc_per_node $GPUS_PER_NODE  --nnodes $NNODES \
     --node_rank $NODE_RANK \
     --master_addr=${MASTER_ADDR:-127.0.0.1} \
     --master_port=$MASTER_PORT \
     web_demo_audio.py \
     --server-port ${SERVER_PORT}