accelerate launch --num_processes 2 --main_process_port 33345 -m lmms_eval \
    --model qwen2_vl_custom_video \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,max_num_frames=384 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_custom_frame256 \
    --output_path ./logs/ \
    --verbosity=DEBUG