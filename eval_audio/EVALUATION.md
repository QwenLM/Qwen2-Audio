## Evaluation

### Dependencies

```bash
apt-get update
apt-get install openjdk-8-jdk
pip install evaluate
pip install sacrebleu==1.5.1
pip install edit_distance
pip install editdistance
pip install jiwer
pip install scikit-image
pip install textdistance
pip install sed_eval
pip install more_itertools
pip install zhconv
```
### ASR

- Data

> LibriSpeech: https://www.openslr.org/12

> Aishell2: https://www.aishelltech.com/aishell_2

> common voice 15: https://commonvoice.mozilla.org/en/datasets

> Fluers: https://huggingface.co/datasets/google/fleurs

```bash
mkdir -p data/asr && cd data/asr

# download audios from above links

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/librispeech_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/aishell2_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_en_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_zh_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_yue_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_fr_eval.jsonl
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/fleurs_asr_zh_eval.jsonl
cd ../..
```

```bash
 for ds in "librispeech" "aishell2" "cv15_en" "cv15_zh" "cv15_yue" "cv15_fr" "fluers_zh"
 do
     python -m torch.distributed.launch --use_env \
         --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
         evaluate_asr.py \
         --checkpoint $checkpoint \
         --dataset $ds \
         --batch-size 20 \
         --num-workers 2
 done
```
### S2TT

- Data

> CoVoST 2: https://github.com/facebookresearch/covost

```bash
mkdir -p data/st && cd data/st

# download audios from https://commonvoice.mozilla.org/en/datasets

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/covost2_eval.jsonl

cd ../..
```
- Evaluate
```bash
ds="covost2"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_st.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### SER
- Data
> MELD: https://affective-meld.github.io/



```bash
mkdir -p data/ser && cd data/ser

# download MELD datasets from above link

# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/meld_eval.jsonl


cd ../..
```

- Evaluate

```bash
ds="meld"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_emotion.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```


### VSC
- Data
> VocalSound: https://github.com/YuanGongND/vocalsound


```bash
mkdir -p data/vsc && cd data/vsc

# download dataset from the above link
# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/vocalsound_eval.jsonl


cd ../..
```

- Evaluate

```bash
ds="vocalsound"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_aqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### AIR-BENCH
- Data
> AIR-BENCH: https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset

```bash
mkdir -p data/airbench && cd data/airbench

# download dataset from the above link
# download converted files
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/airbench_level_3_eval.jsonl


cd ../..
```

```bash
ds="airbench_level3"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_chat.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### Acknowledgement

Part of these codes are borrowed from [Whisper](https://github.com/openai/whisper) , [speechio](https://github.com/speechio/chinese_text_normalization), thanks for their wonderful work.