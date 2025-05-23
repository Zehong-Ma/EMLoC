# Baseline
accelerate launch --num_processes 1 --main_process_port 37345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,max_num_frames=8 \
    --tasks youcook2_val_fewshot \
    --batch_size 1 \
    --log_samples \
    --seed 0,1234,1234,42 \
    --log_samples_suffix qwen2_vl_baseline \
    --output_path ./logs/ \
    --verbosity=DEBUG

## MLoC
# accelerate launch --num_processes 1 --main_process_port 37345 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,max_num_frames=8,num_fewshot=20 \
#     --tasks youcook2_val_fewshot \
#     --batch_size 1 \
#     --log_samples \
#     --seed 0,1234,1234,42 \
#     --log_samples_suffix qwen2_vl_fewshot20 \
#     --output_path ./logs/ \
#     --verbosity=DEBUG

# EMLoC
accelerate launch --num_processes 1 --main_process_port 37345 -m lmms_eval \
    --model qwen2_vl_custom \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,max_num_frames=8,num_fewshot=20 \
    --tasks youcook2_val_fewshot \
    --batch_size 1 \
    --log_samples \
    --seed 0,1234,1234,42 \
    --log_samples_suffix qwen2_vl_custom_fewshot \
    --output_path ./logs/ \
    --verbosity=DEBUG