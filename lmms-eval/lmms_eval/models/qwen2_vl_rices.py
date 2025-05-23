import open_clip
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
from tqdm import tqdm
import numpy as np

def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

# 定义 FewShotDataset 用于 RICES
class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, task_dict, task, split, doc_to_visual):
        import random
        random.seed(42)
        if "imagenet" not in task:
            sel_ctx = random.sample([i_ for i_ in range(len(task_dict[task][split]))], 20) # sample k shot
            self.task_dict = [task_dict[task][split][ind] for ind in sel_ctx]
        self.task_dict = task_dict[task][split]
        self.doc_to_visual = doc_to_visual

    def __len__(self):
        return len(self.task_dict)

    def __getitem__(self, idx):
        item = self.task_dict[idx]
        image = self.doc_to_visual(item)[0]
        return {"image": image}

class RICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # 加载模型和处理器
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        # 预计算特征
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )
        with torch.no_grad():
            for batch in tqdm(loader, desc="Precomputing features for RICES"):
                batch = batch["image"]
                inputs = torch.stack([self.image_processor(image) for image in batch]).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())
        return torch.cat(features)

    def find(self, batch, num_examples):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(self.device)
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()
            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)
            similarity = (query_feature @ self.features.T).squeeze()
            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]
        return [[self.dataset[i] for i in reversed(row)] for row in indices]

import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import os
from PIL import Image
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; 请通过 pip install qwen-vl-utils 安装")
import time

@register_model("qwen2_vl_rices")
class Qwen2_VL_RICES(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_pixels: int = 64*28*28,
        min_pixels: int = 64*28*28,
        max_num_frames: int = 8,
        num_fewshot: int = 0,
        fewshot_split: str = "fewshot",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"未预期的 kwargs: {kwargs}"

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
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=self.device_map
            ).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.k_shot = num_fewshot
        self.fewshot_split = fewshot_split
        self.rices_dict = {}  # 新增：存储每个任务的 RICES 实例

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"使用 {accelerator.num_processes} 个设备进行数据并行")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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
        raise NotImplementedError("Qwen2_VL 未实现 Loglikelihood")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="模型响应中")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        useful_len = len(requests[0].arguments)

        fewshot_ctx_feat = {}  # 用于存储上下文特征

        for chunk in chunks:
            if useful_len == 6:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            else:
                contexts, all_gen_kwargs, doc_to_visual, doc_to_text, doc_to_answer, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            # 如果 k_shot > 0，则使用 RICES 检索相关上下文
            if self.k_shot > 0:
                
                # 初始化 RICES 实例（如果尚未初始化）
                if task not in self.rices_dict:
                    start_adapt = time.time()
                    fewshot_dataset = FewShotDataset(self.task_dict, task, self.fewshot_split, doc_to_visual[0])
                    self.rices_dict[task] = RICES(
                        fewshot_dataset,
                        device=self.device,
                        batch_size=10
                    )
                    end_adapt = time.time()
                    print("rices construct embeddings time: ", end_adapt-start_adapt)
                rices = self.rices_dict[task]

                # 计算批次图像的查询特征
                inputs = torch.stack([rices.image_processor(image) for image in visuals]).to(self.device)
                with torch.no_grad():
                    query_features = rices.model.encode_image(inputs)
                    query_features /= query_features.norm(dim=-1, keepdim=True)
                similarity = query_features @ rices.features.T  # (B, N)
                max_similarity, _ = similarity.max(dim=0)  # (N,)
                _, top_indices = max_similarity.topk(min(self.k_shot, len(rices.features)))
                selected_indices = top_indices.cpu().tolist()
                selected_examples = [self.task_dict[task][self.fewshot_split][idx] for idx in selected_indices]

                # 构建 ctx_msg
                ctx_msg = []
                for example in selected_examples:
                    ctx_visual = doc_to_visual[0](example)
                    ctx_text = doc_to_text[0](example)
                    ctx_ans = doc_to_answer[0](example)
                    if isinstance(ctx_visual, Image.Image):
                        ctx_msg.append({"role": "user", "content": [{"type": "image", "image": ctx_visual}, {"type": "text", "text": ctx_text}]})
                    elif isinstance(ctx_visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in ctx_visual):
                        image_content = [{"type": "image", "image": v} for v in ctx_visual]
                        ctx_msg.append({"role": "user", "content": image_content + [{"type": "text", "text": ctx_text}]})
                    else:
                        ctx_msg.append({"role": "user", "content": [{"type": "text", "text": ctx_text}]})
                    ctx_msg.append({"role": "assistant", "content": [{"type": "text", "text": ctx_ans}]})
                # print(f"使用 RICES 从 {self.fewshot_split} 分割中检索到的上下文：", ctx_msg)
            else:
                ctx_msg = []

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"gen_kwargs['until'] 应为 str 或 list 类型，得到 {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            messages = []
            for i, context in enumerate(contexts):
                message = []
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv")):
                        visual = os.path.expanduser(visual)
                        message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                        image_content = [{"type": "image", "image": v} for v in visual]
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                ctx_msg.extend(message)
                messages.append(ctx_msg)

          
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

            image_inputs, video_inputs = process_vision_info(messages, self.max_num_frames)
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device if self.device_map != "auto" else "cuda")

            gen_kwargs.setdefault("max_new_tokens", 128)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            pad_token_id = self.tokenizer.pad_token_id

          
            custom_kv = None
            custom_kv_pos_offset = 0

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

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: 实现多轮生成")