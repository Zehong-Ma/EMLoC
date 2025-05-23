import base64
from copy import deepcopy
from io import BytesIO
from typing import List, Optional, Tuple, Union
import os

import decord
import numpy as np
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration, BatchEncoding

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord
import time

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_vl_custom_video")
class Qwen2_VL_Custom_Video(lmms):
    """
    Qwen2_VL Model
    "https://github.com/QwenLM/Qwen2-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_pixels: int = 144*28*28,
        min_pixels: int = 144*28*28,
        max_num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.bfloat16, 
                                                                          attn_implementation="sdpa",
                                                                          device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU, # support NPU
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        all_time = 0 
        all_llm_time = 0
        all_video_read = 0
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        total_compressed_lens, total_ori_lens = 0, 0
        iter_ind = 0
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
            time_start_reading = time.time()
            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):
                # print(context)
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                        # max_pixels = height * width
                        message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        for v in visual:
                            image_content.append({"type": "image", "image": v})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
            
            
            
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages, self.max_num_frames)
            # if video_inputs is not None:
            #     total_frames = video_inputs[0].shape[0]
            #     indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
            #     # Append the last frame index if not already included
            #     if total_frames - 1 not in indices:
            #         indices = np.append(indices, total_frames - 1)
            #     video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            time_end_reading = time.time()
            read_video_time = time_end_reading-time_start_reading
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=None,
                video_grid_thw=inputs["video_grid_thw"],
                attention_mask=inputs["attention_mask"],
            )
            
            vid_end_ind = torch.nonzero(inputs["input_ids"]==151653)[-1][-1]+1
            # split into context and question span
            question_inputs = BatchEncoding({
                'icl_input_ids': inputs["input_ids"][:, vid_end_ind:], 
                'icl_attention_mask': inputs["attention_mask"][:, vid_end_ind:],
            })
    
            final_inputs = BatchEncoding({
                'input_ids': inputs["input_ids"][:, vid_end_ind:], 
                'attention_mask': inputs["attention_mask"][:, vid_end_ind:],
            })
            
            clip_frames = 64
            vclip_num = (self.max_num_frames//clip_frames)
            vid_st_ind = torch.nonzero(inputs["input_ids"]==151652)[-1][-1]+1
            split_pixels = torch.chunk(inputs["pixel_values_videos"], chunks=vclip_num, dim=0)
            split_video_grid_thw = deepcopy(inputs["video_grid_thw"])
            split_video_grid_thw[:, 0] = clip_frames//2 # 8 frame into 4 with temporal window 2
            st_ind = 0
            
            kv_cache = None
            custom_kv_pos_offset = 0
            time_visuals = 0
            for iter_num in range(vclip_num):
                end_ind = st_ind+inputs["video_grid_thw"][0].prod()//vclip_num//4
                if iter_num == 0:
                    end_ind += vid_st_ind
                if iter_num ==(vclip_num-1):
                    end_ind = vid_end_ind
                context_inputs = BatchEncoding({
                    'input_ids': inputs["input_ids"][:, st_ind:end_ind], 
                    'attention_mask': inputs["attention_mask"][:, st_ind:end_ind],
                    "position_ids": position_ids[:, :, st_ind:end_ind],
                    'pixel_values_videos': split_pixels[iter_num], 
                    'video_grid_thw': split_video_grid_thw,
                })
                st_ind = end_ind
                with torch.no_grad():
                    kv_cache, custom_kv_pos_offset, kv_avg_len, time_visual = self.model.video_context_organize(
                        **context_inputs, **question_inputs, use_cache=True, 
                        custom_kv=kv_cache, custom_kv_pos_offset=0)
                    time_visuals += time_visual
                    # print("pos_offset", custom_kv_pos_offset)            
            # torch.cuda.empty_cache()
            if self._world_size>1:
                kv_avg_len = torch.tensor([int(kv_avg_len)], dtype=torch.int64).to(self.device)
                ori_len = torch.tensor([kv_cache.get_past_seq_len()], dtype=torch.int64).to(self.device)
                compressed_lens = self.accelerator.gather(kv_avg_len) 
                ori_lens = self.accelerator.gather(ori_len)
                total_compressed_lens += compressed_lens.sum().item()
                total_ori_lens += ori_lens.sum().item()
            else:
                total_compressed_lens += kv_avg_len
                total_ori_lens += kv_cache.get_past_seq_len()
            # print("final total compressed length: ", kv_avg_len, "/", kv_cache.get_past_seq_len())
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 16
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id
            # print(custom_kv_pos_offset, position_ids[:,:,vid_end_ind].max())
            assert custom_kv_pos_offset==position_ids[:,:,vid_end_ind].max()
            cont = self.model.generate(
                **final_inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                custom_kv=kv_cache,
                custom_kv_pos_offset=custom_kv_pos_offset,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(final_inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            # print(answers)
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            time_end_generate = time.time()
            time_llm_generate = time_end_generate - time_end_reading - time_visuals
            all_time += (time_end_generate-time_start_reading)
            all_llm_time += time_llm_generate
            all_video_read += time_end_reading-time_start_reading
            if self._rank == 0:
                pbar.set_postfix({
                    "com_len": total_compressed_lens,
                    "ori_len": total_ori_lens,
                    "compressed_ratio": (total_compressed_lens/total_ori_lens),
                    "llm_time_ratio": all_llm_time/all_time,
                    "avg_read_video": all_video_read/(iter_ind+1)
                })
            # reorder this group of results back to original unsorted form
            iter_ind+=1
            # if iter_ind>10:
            #     break
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
