import json
import os
import sys
import math
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

import librosa
from io import BytesIO
from urllib.request import urlopen

from peft import get_peft_model
from peft import LoraConfig, TaskType

import torch
from torch.utils.data import DataLoader
from datasets import IterableDataset

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType

import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    SchedulerType,
    get_scheduler,
)


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="qwen2-audio",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--steps_per_print",
        type=int,
        default=1,
        help="Number of steps before printing the loss.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for the model and tokenizer.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Use low CPU memory usage for the model.",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use FlashAttention for the model.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default=SchedulerType.LINEAR,
        help="The learning rate scheduler type to use.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Number of steps before saving the model.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use lora to finetune.",
    )
    return parser.parse_args()


def toy_data():
    conversation = [
        {
            "role": "system", "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
                }
            ]
        },
        {
            "role": "assistant", "content": "Yes, the speaker is female and in her twenties."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"
                }
            ]
        }
    ]
    conversation1 = [
        {
            "role": "system", "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                },
                {
                    "type": "text", "text": "What's that sound?"
                },
            ]
        },
        {
            "role": "assistant", "content": "It is the sound of glass shattering."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What can you do when you hear that?"},
            ]
        },
        {
            "role": "assistant",
            "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property."
        },
        {
            "role": "user", "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
                },
                {"type": "text", "text": "What does the person say?"},
            ]
        }
    ]
    while True:
        if random.random() < 0.5:
            yield {"conversations": conversation}
        else:
            yield {"conversations": conversation1}


def main(args):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator_log_kwargs = {"dispatch_batches": False}

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs
    )

    def init_dataloader(processor):
        def _func(batch):
            # copy from `https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct`
            conversations = batch["conversations"]

            text = [
                processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    tokeni_in_conversationze=False
                )
                for conversation in conversations
            ]

            audios, audio_num_for_each_conversation = [], []
            for conversation in conversations:
                audio_num = 0
                for message in conversation:
                    if isinstance(message["content"], list):
                        for ele in message["content"]:
                            if ele["type"] == "audio":
                                audios.append(
                                    librosa.load(
                                        BytesIO(
                                            urlopen(ele['audio_url']).read()
                                        ),
                                        sr=processor.feature_extractor.sampling_rate)[0]
                                )
                                audio_num += 1
                audio_num_for_each_conversation.append(audio_num)

            inputs = processor(
                text=text,
                audios=audios if audios else None,
                return_tensors="pt",
                padding=True
            )

            # Split the tensors for each conversation, make sure the dataset is iterable
            inputs["feature_attention_mask"] = [
                x for x in torch.split(
                    inputs["feature_attention_mask"],
                    audio_num_for_each_conversation, dim=0)
            ]
            inputs["input_features"] = [
                x for x in torch.split(
                    inputs["input_features"],
                    audio_num_for_each_conversation,
                    dim=0
                )
            ]
            logger.warning(
                "We automatically learn from all tokens except for `audio` in the conversation. If you want to learn about a specific `role` or `content`, please modify the code accordingly."
            )
            # Qwen2AudioForConditionalGeneration will automatically shift the input_ids for you
            inputs["labels"] = inputs["input_ids"]
            return inputs

        # Load dataset
        dataset = IterableDataset.from_generator(toy_data)
        dataset = dataset.map(
            _func,
            batched=True,
            remove_columns=["conversations"],
            batch_size=2
        )

        def collate_fn(batch):
            flatten_batch = defaultdict(list)
            for k in batch[0]:
                for instance in batch:
                    if isinstance(instance[k], list):
                        flatten_batch[k] += instance[k]
                    else:
                        flatten_batch[k].append(instance[k])
            return {
                k: torch.cat(v, dim=0)
                if k in ["feature_attention_mask", "input_features"] else torch.stack(v)
                for k, v in flatten_batch.items()
            }

        dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )
        return dataloader

    accelerator.state.deepspeed_plugin.deepspeed_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config[
        'gradient_accumulation_steps'] = args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config[
        'steps_per_print'] = args.steps_per_print

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        # Qwen2AudioForConditionalGeneration can not support `flash_attention` but we keep it here for demonstration
        attn_implementation="flash_attention_2" if args.flash_attention else None,
        torch_dtype=config.torch_dtype
    )

    if args.lora:
        logger.info("Use lora to finetune...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(processor.tokenizer) > embedding_size:
        model.resize_token_embeddings(len(processor.tokenizer))

    # Prepare the dataloader
    train_dataloader = init_dataloader(processor)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate
    )

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Num processes: {accelerator.num_processes}")
    logger.info(f"  Process index: {accelerator.process_index}")

    completed_steps = 0

    for _, batch in enumerate(train_dataloader):
        model.train()

        with accelerator.accumulate(model):
            # Move the batch to the device (should be done by the accelerator)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.device == torch.device("cpu"):
                    batch[k] = v.cuda()

            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each step
            local_loss = loss.detach().float()
            logger.info(
                f"Steps = {completed_steps + 1}, Local loss = {local_loss}...")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            completed_steps += 1

        if args.output_dir is not None and completed_steps % args.save_interval == 0:
            accelerator.wait_for_everyone()
            output_dir = os.path.join(
                args.output_dir,
                f"checkpoint_{completed_steps}"
            )
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                processor.save_pretrained(output_dir)

        if completed_steps >= args.max_train_steps:
            return


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    main(args)
