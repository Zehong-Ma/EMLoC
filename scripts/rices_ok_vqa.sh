accelerate launch --num_processes 4 --main_process_port 32545 -m lmms_eval \
    --model qwen2_vl_rices \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,num_fewshot=5 \
    --tasks ok_vqa_val2014_fewshot \
    --batch_size 1 \
    --log_samples \
    --seed 0,1234,1234,42 \
    --log_samples_suffix qwen2_vl_fewshot20 \
    --output_path ./logs/ \
    --verbosity=DEBUG