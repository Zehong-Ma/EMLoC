## Baseline: Zero shot evaluation, default num_fewshot=0
# accelerate launch --num_processes 1 --main_process_port 32345 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False \
#     --tasks illusionvqa_soft_localization_fewshot \
#     --batch_size 4 \
#     --log_samples \
#     --seed 0,1234,1234,42 \
#     --log_samples_suffix qwen2_vl_baseline \
#     --output_path ./logs/ \
#     --verbosity=DEBUG


## MLoC
# accelerate launch --num_processes 1 --main_process_port 38345 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,num_fewshot=20 \
#     --tasks illusionvqa_soft_localization_fewshot \
#     --batch_size 4 \
#     --log_samples \
#     --seed 0,1234,1234,42 \
#     --log_samples_suffix qwen2_vl_MLoC \
#     --output_path ./logs/ \
#     --verbosity=DEBUG

## EMLoC max_pixels=min_pixels=64*28*28
accelerate launch --num_processes 1 --main_process_port 38345 -m lmms_eval \
    --model qwen2_vl_custom \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,num_fewshot=200,fewshot_split=train,max_pixels=50176,min_pixels=50176 \
    --tasks imagenet100_fewshot \
    --batch_size 1 \
    --log_samples \
    --seed 0,1234,1234,42 \
    --log_samples_suffix qwen2_vl_EMLoC \
    --output_path ./logs/ \
    --verbosity=DEBUG
