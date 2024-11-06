import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
import requests
import csv
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from sklearn.metrics import accuracy_score
import os


ds_collections = {
    'meld': {'path': '/home/rsingh57/audio-test/mutox-dataset/non_toxic'}
}


class AudioDataset(torch.utils.data.Dataset):

    # def __init__(self, ds):
    #     path = ds['path']
    #     self.datas = open(path).readlines()
    def __init__(self, audio_folder):
        self.audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav') or f.endswith('.mp3')]
        self.audio_folder = audio_folder
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_folder, audio_file)
        prompt = f"Transcribe audio file: {audio_file}"
        gt = "Ground truth transcription"  
        return {
            'audio': audio_path,
            'prompt': prompt,
            'source': audio_file,
            'gt': gt
        }

        
# def read_audio(audio_path):
#     if audio_path.startswith("http://") or audio_path.startswith("https://"):
#         inputs = requests.get(audio_path).content
#     else:
#         with open(audio_path, "rb") as f:
#             inputs = f.read()
#     return inputs

def read_audio(audio_path):
    with open(audio_path, "rb") as f:
        inputs = f.read()
    return inputs



# def collate_fn(inputs, processor):
#     input_texts = [_['prompt'] for _ in inputs]
#     source = [_['source'] for _ in inputs]
#     gt = [_['gt'] for _ in inputs]
#     audio_path = [_['audio'] for _ in inputs]
#     input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
#     inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
#     return inputs, audio_path, source, gt

def collate_fn(inputs, processor):
    input_texts = [item['prompt'] for item in inputs]
    source = [item['source'] for item in inputs]
    gt = [item['gt'] for item in inputs]
    audio_path = [item['audio'] for item in inputs]
    input_audios = [ffmpeg_read(read_audio(path), sampling_rate=processor.feature_extractor.sampling_rate) for path in audio_path]
    inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt

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
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--audio-folder', type=str,  default="/home/rsingh57/audio-test/mutox-dataset/non_toxic")
    parser.add_argument('--dataset', type=str, default='meld')
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
        args.checkpoint, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    
    dataset = AudioDataset(
        audio_folder=args.audio_folder
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    gts, sources, rets, audio_paths = [], [], [], []
    for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader)):
        inputs['input_ids'] = inputs['input_ids'].to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
        output_ids = output_ids[:, inputs.input_ids.size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gts.extend(gt)
        rets.extend(output)
        sources.extend(source)
        audio_paths.extend(audio_path)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        print("Evaluating ...")
        results = []
        for gt, response, source, audio_path in zip(gts, rets, sources, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'source': source,
                'audio_path': audio_path,
            })
        results_file = f'evaluation_results.csv'
        with open(results_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['gt', 'response', 'source', 'audio_path'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {results_file}")
        
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )

    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # model = Qwen2AudioForConditionalGeneration.from_pretrained(
    #     args.checkpoint, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()

    # processor = AutoProcessor.from_pretrained(args.checkpoint)

    # processor.tokenizer.padding_side = 'left'

    # random.seed(args.seed)
    # dataset = AudioDataset(
    #     ds=ds_collections[args.dataset]
    # )
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     sampler=InferenceSampler(len(dataset)),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    #     collate_fn=partial(collate_fn, processor=processor),
    # )

    # gts = []
    # sources = []
    # rets = []
    # audio_paths = []
    # for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader)):
    #     inputs['input_ids'] = inputs['input_ids'].to('cuda')
    #     output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
    #     output_ids = output_ids[:, inputs.input_ids.size(1):]
    #     output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #     gts.extend(gt)
    #     rets.extend(output)
    #     sources.extend(source)
    #     audio_paths.extend(audio_path)

    # torch.distributed.barrier()

    # world_size = torch.distributed.get_world_size()
    # merged_gts = [None for _ in range(world_size)]
    # merged_sources = [None for _ in range(world_size)]
    # merged_responses = [None for _ in range(world_size)]
    # merged_audio_paths = [None for _ in range(world_size)]
    # torch.distributed.all_gather_object(merged_gts, gts)
    # torch.distributed.all_gather_object(merged_sources, sources)
    # torch.distributed.all_gather_object(merged_responses, rets)
    # torch.distributed.all_gather_object(merged_audio_paths, audio_paths)

    # merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    # merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    # merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    # merged_responses = [
    #     _ for _ in itertools.chain.from_iterable(merged_responses)
    # ]

    # if torch.distributed.get_rank() == 0:
    #     print(f"Evaluating {args.dataset} ...")

    #     results = []
    #     for gt, response, source, audio_path in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths):
    #         results.append({
    #             'gt': gt,
    #             'response': response,
    #             'source': source,
    #             'audio_path': audio_path,
    #         })
    #     time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    #     results_file = f'{args.dataset}_{time_prefix}.json'
    #     json.dump(results, open(results_file, 'w'))
    #     results_dict = {}
    #     for item in tqdm(results):
    #         source = item["source"]
    #         results_dict.setdefault(source, []).append(item)

    #     for source in results_dict:
    #         refs, hyps = [], []
    #         bi_refs, bi_hyps = [], []
    #         results_list = results_dict[source]
    #         for result in results_list:
    #             gt = result["gt"]
    #             response = result["response"].lstrip()
    #             refs.append(gt)
    #             hyps.append(response)
    #         score = accuracy_score(refs, hyps)
    #         print(f"{source} ACC_score:", score, len(hyps))

    # torch.distributed.barrier()

    # if torch.distributed.get_rank() == 0:
    #     print(f"Evaluating {args.dataset} ...")
    #     results = []
    #     for gt, response, source, audio_path in zip(gts, rets, sources, audio_paths):
    #         results.append({
    #             'gt': gt,
    #             'response': response,
    #             'source': source,
    #             'audio_path': audio_path,
    #         })

    #     results_file = f'{args.dataset}_mutox_test.csv'
    #     with open(results_file, mode='w', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames=['gt', 'response', 'source', 'audio_path'])
    #         writer.writeheader()
    #         writer.writerows(results)
    #     print(f"Results saved to {results_file}")
