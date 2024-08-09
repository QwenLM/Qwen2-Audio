import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
import requests
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read


ds_collections = {
    'airbench_level3': {'path': 'chat/airbench-level-3.jsonl'}
}


class AudioChatDataset(torch.utils.data.Dataset):

    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        data_idx = data['id']
        query = data['query']

        return {
            'audio': audio,
            'data_idx': data_idx,
            'query': query,
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    text_list = []
    for _ in inputs:
        query = _['query']
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
        text = processor.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            tokenize=False
        )
        text_list.append(text)

    audio_path = [_['audio'] for _ in inputs]
    data_idxs = [_['data_idx'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    inputs = processor(text=text_list, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, data_idxs


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda', torch_dtype='auto', trust_remote_code=True).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioChatDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )


    idxs = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, data_idxs) in tqdm(enumerate(data_loader)):
        inputs['input_ids'] = inputs['input_ids'].to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1,do_sample=False)
        output_ids = output_ids[:, inputs.input_ids.size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets.extend(output)
        audio_paths.extend(audio_path)
        idxs.extend(data_idxs)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_idxs = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_idxs, idxs)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_audio_paths, audio_paths)

    merged_idxs = [_ for _ in itertools.chain.from_iterable(merged_idxs)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for idx, response, audio_path in zip(merged_idxs, merged_responses, merged_audio_paths):
            results.append({
                'idx': idx,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    torch.distributed.barrier()
