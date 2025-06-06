import base64
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
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_vl")
class Qwen2_VL(lmms):
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
        max_pixels: int = 256*28*28, #int = 512*28*28, #int = 12845056,
        min_pixels: int = 256*28*28,
        max_num_frames: int = 8,
        num_fewshot: int = 0,
        fewshot_split: str="fewshot",
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
  
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.k_shot = num_fewshot
        self.fewshot_split = fewshot_split
        self.image_path_ids = 0
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
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

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        useful_len = len(requests[0].arguments)
        
        fewshot_ctx, fewshot_ctx_feat = {}, {}
        fewshot_json, eval_json = [], []
        for chunk in chunks:
            if useful_len == 6:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            else:
                contexts, all_gen_kwargs, doc_to_visual, doc_to_text, doc_to_answer, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            
            visuals = self.flatten(visuals)
            
            eval_ans = [doc_to_answer[0](self.task_dict[task][split][ids]) for ids in doc_id]
            if task not in fewshot_ctx.keys() and self.k_shot>0:
                # generate context for the task
                import random
                random.seed(42)
                sel_ctx = random.sample([i_ for i_ in range(len(self.task_dict[task][self.fewshot_split]))], self.k_shot) # sample k shot
                ctx_text = [doc_to_text[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                ctx_ans = [doc_to_answer[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                
                ctx_visuals = [doc_to_visual[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                ctx_visuals = self.flatten(ctx_visuals)  
                print(f"using {self.fewshot_split} split as fewshot demonstrations")
                ctx_msg = []
                for ind in range(len(sel_ctx)):
                    if len(ctx_visuals) > 0:
                        visual = ctx_visuals[ind] if ind < len(ctx_visuals) else None
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                            visual = os.path.expanduser(visual)
                            # max_pixels = height * width
                            ctx_msg.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": ctx_text[ind]}]})
                        elif isinstance(visual, Image.Image):  # Single image
                            ctx_msg.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": ctx_text[ind]}]})
                        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                            image_content = []
                            for v in visual:
                                image_content.append({"type": "image", "image": v})
                            ctx_msg.append({"role": "user", "content": image_content + [{"type": "text", "text": ctx_text[ind]}]})
                        else:
                            ctx_msg.append({"role": "user", "content": [{"type": "text", "text": ctx_text[ind]}]})
                        
                        # save images and llamafact datasets jsons
                        # file_path = f"/nfs/mzh/LLM/LongICL/dataset_IllusionVQA/images/{self.image_path_ids}.png"
                        # visual.save(file_path)
                        # self.image_path_ids +=1
                        
                        # fewshot_example = {
                        #     "messages":[
                        #         {
                        #             "content": f"<image>{ctx_text[ind]}",
                        #             "role": "user"
                        #         },
                        #         {
                        #             "content": f"{ctx_ans[ind]}",
                        #             "role": "assistant"
                        #         }
                        #     ],
                        #     "images": [
                        #         file_path
                        #     ]
                        # }
                        # fewshot_json.append(fewshot_example)
                        
                    else:
                        ctx_msg.append({"role": "user", "content": [{"type": "text", "text": ctx_text[ind]}]})
                    ctx_msg.append({"role": "assistant", "content": [{"type": "text", "text": ctx_ans[ind]}]})
                fewshot_ctx[task] = ctx_msg
                print(ctx_msg)
            elif self.k_shot>0:
                ctx_msg = fewshot_ctx[task]
                
            else:
                ctx_msg = None
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

            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                
                # message = [{"role": "system", "content": "You are a helpful assistant."}]
                message = []
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                        visual = os.path.expanduser(visual)
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
                
                # file_path = f"/nfs/mzh/LLM/LongICL/dataset_IllusionVQA/images/{self.image_path_ids}.png"
                # visuals[i].save(file_path)
                # self.image_path_ids +=1
                
                # eval_example = {
                #     "messages":[
                #         {
                #             "content": f"<image>{context}",
                #             "role": "user"
                #         },
                #         {
                #             "content": f"{eval_ans[i]}",
                #             "role": "assistant"
                #         }
                #     ],
                #     "images": [
                #         file_path
                #     ]
                # }
                # eval_json.append(eval_example)
            # print(message)
            if ctx_msg is not None:
                ctx_texts = [self.processor.apply_chat_template(ctx_msg, tokenize=False, add_generation_prompt=True)[:-22]]
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)[58:] for msg in messages]
                ctx_image_inputs, ctx_video_inputs = process_vision_info([ctx_msg], self.max_num_frames)
                ctx_inputs = self.processor(text=ctx_texts, images=ctx_image_inputs, videos=ctx_video_inputs, padding=True, return_tensors="pt")
                if self.device_map == "auto":
                    ctx_inputs = ctx_inputs.to("cuda")
                else:
                    ctx_inputs = ctx_inputs.to(self.device)
            else: # zero shot
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
            
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
                
                
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id
            
            if ctx_msg is not None:
                if task not in fewshot_ctx_feat.keys():
                    with torch.no_grad():
                        custom_kv, custom_kv_pos_offset = self.model.forward_context(
                            **ctx_inputs
                        )
                    fewshot_ctx_feat[task] = (custom_kv, custom_kv_pos_offset)
                else:
                    custom_kv, custom_kv_pos_offset = fewshot_ctx_feat[task]
            else:
                custom_kv = None
                custom_kv_pos_offset = 0
                # print("context_length", custom_kv.get_past_seq_len())
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                custom_kv=custom_kv,
                custom_kv_pos_offset=custom_kv_pos_offset,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
                # print(ans)
                # print(parse_float_sequence_within(ans))
            # print(answers)
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # import json
        # import os

        # with open("/nfs/mzh/LLM/LongICL/dataset_IllusionVQA/illusion_vqa_fewshot_adapt.json", "w") as wf1:
        #     json.dump(fewshot_json, wf1, indent=1)
        # with open("/nfs/mzh/LLM/LongICL/dataset_IllusionVQA/illusion_vqa_eval.json", "w") as wf2:
        #     json.dump(eval_json, wf2, indent=1) 
        
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    import re
    # Define the regex pattern to find the first instance of four floats within square brackets
    # pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    pattern = r"(?:\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\))|(?:\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\])|(?:Top-left.*?x:\s*(-?\d+(?:\.\d+)?).*?y:\s*(-?\d+(?:\.\d+)?).*?Bottom-right.*?x:\s*(-?\d+(?:\.\d+)?).*?y:\s*(-?\d+(?:\.\d+)?))|(?:Top-left:\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)\s*Bottom-right:\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\))"
    match = re.search(pattern, input_str)
    
    # If a match is found, convert the captured groups into a list of floats
    if match:
        try:
            res = [float(match.group(i)) for i in range(1, 5)]
        except:
            res = [num for num in map(float, match.group(0).strip('[]').split(","))]
        if res[0]>1:
            return res/1000
        else:
            return res
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]