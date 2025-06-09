import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import os

import decord
import numpy as np
import torch
import random
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

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用CUDA，也设置这个
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("qwen2_vl_custom")
class Qwen2_VL_Custom(lmms):
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
        max_pixels: int = 256*28*28,
        min_pixels: int = 256*28*28,
        max_num_frames: int = 32,
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
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.bfloat16, 
                                                                          attn_implementation="sdpa",
                                                                        #   cache_dir="/nfs/mzh/LLM/pretrained_models/",
                                                                          device_map=self.device_map).eval()
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
    
    

    def process_outer_iter(self, context, model, processor, device, outer_kv, custom_kv_pos_offset=0, compressed_len=0):
        context_texts = [
                processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True, add_vision_id=False)[:-22]
                if outer_kv is None else processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True, add_vision_id=False)[58:-22]
        ]
        context_image_inputs, context_video_inputs = process_vision_info([context])
        context_inputs = processor(
            text=context_texts,
            images=context_image_inputs,
            videos=context_video_inputs,
            padding=False,
            return_tensors="pt",
        )
        labels = -100*torch.ones_like(context_inputs["input_ids"])
        label_starts = torch.nonzero(context_inputs["input_ids"]==77091)+1
        label_ends = torch.nonzero(context_inputs["input_ids"]==151645)[2::2] if outer_kv is None else torch.nonzero(context_inputs["input_ids"]==151645)[1::2]
        for label_start_, label_end_ in zip(label_starts[:,1], label_ends[:,1]):
            labels[0, label_start_:label_end_+1] = context_inputs["input_ids"].squeeze(0)[label_start_:label_end_+1]
        
        context_inputs["labels"] = labels
        context_inputs = context_inputs.to(device)
        
        # sep inputs
        sep_context = [context[i:i+2] for i in range(0, len(context), 2)]
        sep_context_texts = [
                processor.apply_chat_template(sep_context_, tokenize=False, add_generation_prompt=True, add_vision_id=False)[58:-22]
                for sep_context_ in sep_context
        ]
        sep_context_inputs = processor(
            text=sep_context_texts,
            images=context_image_inputs,
            videos=context_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        sep_labels = -100*torch.ones_like(sep_context_inputs["input_ids"])
        sep_label_starts = torch.nonzero(sep_context_inputs["input_ids"]==77091).squeeze(0)+1 # only label
        # sep_label_starts = torch.nonzero(sep_context_inputs["input_ids"]==151653).squeeze(0)+1
        for ind in range(sep_labels.shape[0]):
            sep_labels[ind, sep_label_starts[ind, 1]:-1] = sep_context_inputs["input_ids"][ind, sep_label_starts[ind, 1]:-1]
        sep_context_inputs["labels"] = sep_labels
        sep_context_inputs = sep_context_inputs.to(device)
        sep_context_inputs = {"icl_"+k: v for k, v in sep_context_inputs.items()}
        
        with torch.no_grad():
            kv_cache, custom_kv_pos_offset, kv_avg_len, all_img_num, all_ques_num, all_ans_num, all_img_keep, all_ques_keep, all_ans_keep = model.context_organize_pre(**context_inputs, **sep_context_inputs, use_cache=True, 
                                                                                custom_kv=outer_kv, custom_kv_pos_offset=custom_kv_pos_offset)
        torch.cuda.empty_cache()
        compressed_len = kv_avg_len
        return kv_cache, custom_kv_pos_offset, compressed_len, all_img_num, all_ques_num, all_ans_num, all_img_keep, all_ques_keep, all_ans_keep
    
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
        for chunk in chunks:
            if useful_len == 6:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            else:
                contexts, all_gen_kwargs, doc_to_visual, doc_to_text, doc_to_answer, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            if task not in fewshot_ctx.keys() and self.k_shot>0:
                # generate context for the task
                if "imagenet" in task.lower():
                    set_seed(0)
                    sel_ctx =[i for i in range(self.k_shot)]
                    # print(sel_ctx)
                else:
                    import random
                    random.seed(42)
                    sel_ctx = random.sample([i_ for i_ in range(len(self.task_dict[task][self.fewshot_split]))], self.k_shot) # sample k shot
                ctx_text = [doc_to_text[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                # print(doc_to_answer)
                ctx_ans = [doc_to_answer[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                ctx_visuals = [doc_to_visual[0](self.task_dict[task][self.fewshot_split][ctx_]) for ctx_ in sel_ctx]
                ctx_visuals = self.flatten(ctx_visuals)
                print(f"using {self.fewshot_split} split as fewshot demonstrations")
                
                ctx_msg = []
                for ind in range(len(sel_ctx)):
                    if len(ctx_visuals) > 0:
                        visual = ctx_visuals[ind] if ind < len(ctx_visuals) else None
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            visual = os.path.expanduser(visual)
                            vr = decord.VideoReader(visual, ctx=decord.cpu(), num_threads=16)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
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
                    else:
                        ctx_msg.append({"role": "user", "content": [{"type": "text", "text": ctx_text[ind]}]})
                    ctx_msg.append({"role": "assistant", "content": [{"type": "text", "text": ctx_ans[ind]}]})
                fewshot_ctx[task] = ctx_msg
                # print(ctx_msg)
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
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        visual = os.path.expanduser(visual)
                        vr = decord.VideoReader(visual, ctx=decord.cpu(), num_threads=16)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
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

            if ctx_msg is not None:
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)[58:] for msg in messages]
                
                if task not in fewshot_ctx_feat.keys():
                    img_num, ques_num, ans_num, img_keep, ques_keep, ans_keep = 0, 0, 0, 0, 0, 0
                    samples_per_iter = 20 if "imagenet" in task.lower() else 4 #if len(ctx_msg)>8*2 else 16
                    contexts_split = [ctx_msg[i*(samples_per_iter*2):(i+1)*(samples_per_iter*2)]  
                                    for i in range(len(ctx_msg)//(samples_per_iter*2)+int(len(ctx_msg)%(samples_per_iter*2)>0))]
                    outer_kv = None
                    kv_offset = 0
                    compressed_len = 0
                    device = "cuda" if self.device_map == "auto" else self.device
                    for ind, sub_context in enumerate(contexts_split):
                        outer_kv, kv_offset, compressed_len,all_img_num, all_ques_num, all_ans_num, all_img_keep, all_ques_keep, all_ans_keep = self.process_outer_iter(sub_context, self.model, self.processor, device, outer_kv, kv_offset, compressed_len)
                        img_num += all_img_num.item()
                        ques_num += all_ques_num.item()
                        ans_num += all_ans_num.item()
                        total = img_num + ques_num + ans_num
                        # print("pruning all types, img, ques, ans", img_num, ques_num, ans_num, img_num/total, ques_num/total, ans_num/total)
                        img_keep +=all_img_keep.item()
                        ques_keep +=all_ques_keep.item()
                        ans_keep +=all_ans_keep.item()
                        total1 = img_keep+ques_keep+ans_keep
                        # print("keep all types, img, ques, ans", img_keep, ques_keep, ans_keep, img_keep/total1, ques_keep/total1, ans_keep/total1)
        
                    print("final total compressed length: ", compressed_len, "/", outer_kv.get_past_seq_len())
                    fewshot_ctx_feat[task] = (outer_kv, kv_offset)
                else:
                    outer_kv, kv_offset = fewshot_ctx_feat[task]  
            else:
                outer_kv = None
                kv_offset = 0        
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            
            image_inputs, video_inputs = process_vision_info(messages, self.max_num_frames)
            
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
                custom_kv=outer_kv,
                custom_kv_pos_offset=kv_offset,
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

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
