# [ICML 2025] Efficient Multi-modal Long Context Learning for Training-free Adaptation

<div class="is-size-5 publication-authors", align="center",>
            <span class="author-block">
              <a href="https://liewfeng.github.io" target="_blank">Zehong Ma</a><sup>1</sup><sup></sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ" target="_blank">Shiliang Zhang</a><sup>1</sup><sup>*</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://jeffwang987.github.io" target="_blank">Longhui Wei</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://weilllllls.github.io" target="_blank">Qi Tian</a><sup>2</sup> <sup>3</sup>,&nbsp;
          </div>

<div class="is-size-5 publication-authors", align="center">
            <span class="author-block"><sup>1</sup>Peking University,&nbsp;</span>
            <span class="author-block"><sup>2</sup>Huawei Inc.,</span>
            <span class="author-block"><sup>3</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ).</span>
            <br>
          </div>


<div class="is-size-5 publication-authors", align="center">
            (* CorresCorresponding author.)
          </div>

<!-- <h5 align="center"> -->
<!-- 
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2411.19108)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.19108-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19108) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://liewfeng.github.io/TeaCache/) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE) 
[![github](https://img.shields.io/github/stars/LiewFeng/TeaCache.svg?style=social)](https://github.com/LiewFeng/TeaCache/) -->

<!-- </h5> -->


<!-- ![visualization](./assets/tisser.png) -->

## 🫖 Introduction 
We introduce Efficient Multi-Modal Long Context Learning (EMLoC), a novel training-free method that embeds many demonstration examples directly into the model input. EMLoC offers a more efficient, flexible, and scalable solution for task adaptation. By adaptively pruning tokens at each layer under a Jensen-Shannon divergence constraint, our method achieves a dramatic reduction in inference complexity without sacrificing performance.
 <!-- For more details and results, please visit our [project page](https://liewfeng.github.io/TeaCache/). -->

## 🔥 Latest News 
- **If you like our project, please give us a star ⭐ on GitHub for the latest update.**
- [2025/5/4] 🎉 Release the paper and code of EMLoC.
  
## 🎉 Supported Models 
**Multi-modal Large Language Models**
- [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL)
- Other models will be supported soon!
<!-- - [InternVL3](https://github.com/OpenGVLab/InternVL/tree/main) -->

## 🤖 Instructions for EMLoC
### Environments
This work is built based on [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [transformers](https://github.com/huggingface/transformers). Thanks for their contributions! We modify the `modeling_qwen2_vl.py` and `cache_utils.py` in transformers. Besides, we modify the lmms-eval to support multi-modal in-context learning and Ascend 910B. More details can be seen in the code.

```
git clone 
cd EMLoC
conda create -y -n emloc python=3.10
conda activate emloc
pip install -r requirements

## install torch-npu to support Ascend 910B
# pip install torch-npu==2.4.0 
```
### Data Preparation

+ ImageNet1k: `train` and `val` should be in the root dir of imagenet1k.
```
ln -s /path/to/imagenet1k/ ./data/imagenet1k
```

+ Other datasets: lmms-eval will automatically download and spilt them into fewshot set and validation set.

### Run EMLoC on different datasets.
```shell
# ImageNet
sh ./scripts/EMLoC_imagenet.sh

# illusionVQA
sh ./scripts/EMLoC_illusionVQA.sh

# mmerealworld
sh ./scripts/EMLoc_mmerealworld_lite.sh

# More datasets please see ./scripts/

```

<!-- ## 💐 Acknowledgement 
This repository is built based on [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [transformers](https://github.com/huggingface/transformers). Thanks for their contributions! -->

<!-- ## 🔒 License 
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file. -->
<!-- ## 📖 Citation 
If you find EMLoC is useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry. -->

