accelerate launch --num_processes 1 --main_process_port 32845 -m lmms_eval \
    --model qwen2_vl_rices \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,num_fewshot=40,fewshot_split=train \
    --tasks imagenet100_fewshot \
    --batch_size 1 \
    --log_samples \
    --seed 0,1234,1234,42 \
    --log_samples_suffix qwen2_vl_RICES_fewshot20 \
    --output_path ./logs/ \
    --verbosity=DEBUG