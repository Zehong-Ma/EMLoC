import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, MllamaForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import av
warnings.filterwarnings("ignore")
from PIL import Image
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<|image|>"


@register_model("llama_vision")
class LlamaVision(lmms):
    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-3.2-11B-Vision",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        max_frames_num: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = f"cuda:0"
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.max_frames_num = max_frames_num
        self._model = MllamaForConditionalGeneration.from_pretrained(pretrained, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation, cache_dir="/nfs/mzh/LLM/pretrained_models/")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, cache_dir="/nfs/mzh/LLM/pretrained_models/")
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    # def load_video(self, video_path, max_frames_num):
    #     if type(video_path) == str:
    #         vr = VideoReader(video_path, ctx=cpu(0))
    #     else:
    #         vr = VideoReader(video_path[0], ctx=cpu(0))
    #     total_frame_num = len(vr)
    #     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    #     frame_idx = uniform_sampled_frames.tolist()
    #     spare_frames = vr.get_batch(frame_idx).asnumpy()
    #     return spare_frames  # (frames, height, width, channels)

    def load_video(self, video_path, max_frames_num=8):
        """Optimized PyAV video reading with frame seeking and minimal overhead."""
      
        container = av.open(video_path)
        stream = container.streams.video[0]
        video_fps = stream.average_rate
        if video_fps is None:
            # 可以根据需求：猜一个帧率，或提示用户
            print("警告：无法从视频流中获取 average_rate，默认使用 30fps。")
            video_fps = 30.0
        else:
            video_fps = float(video_fps)

        total_frames = stream.frames
        if not total_frames or total_frames <= 0:
            print("警告：无法从视频流中获取 frames，默认抽帧逻辑将尽力而为。")
            total_frames = 999999
        nframes = max_frames_num
        frame_indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)

        time_base = stream.time_base  # 一般是 Fraction 类型，类似 1/30
        time_base_num = time_base.numerator
        time_base_den = time_base.denominator
        
        frames = []

        for idx in frame_indices:
    
            time_seconds = idx / video_fps
            pts = int(time_seconds * time_base_den / time_base_num)
            container.seek(pts, any_frame=False, backward=True, stream=stream)
            decoded_frames = container.decode(video=0)
            try:
                frame = next(decoded_frames)
            except StopIteration:
                # 如果在视频尾部出现越界，就跳过
                print(f"警告：尝试解码 idx={idx} 时出现 StopIteration。")
                break
            img = Image.fromarray(frame.to_ndarray(format="rgb24")).convert("RGB")
            frames.append(img)
        container.close()
        return frames
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            messages = [{"role": "user", "content": []}]
            images = []

            for visual in visuals:
                if isinstance(visual, str):
                    images = self.load_video(visual, self.max_frames_num)
                    if len(images) <1:
                        continue
                    # frames = self.load_video(visual, self.max_frames_num)
                    # frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                    # images.extend([to_pil_image(frame) for frame in frames])
                elif isinstance(visual, PIL.Image.Image):
                    images.append(visual)

            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            messages[-1]["content"].append({"type": "text", "text": contexts})
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images, prompt, return_tensors="pt").to(self.model.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    temperature=gen_kwargs["temperature"],
                    do_sample=gen_kwargs["do_sample"],
                )
                output = output[:, inputs["input_ids"].shape[-1] :]
                print(self.processor.decode(output[0]))
                res.append(self.processor.decode(output[0]))

            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
