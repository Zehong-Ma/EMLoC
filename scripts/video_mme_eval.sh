export CUDA_VISIBLE_DEVICES=1,2
accelerate launch --num_processes 2 --main_process_port 32375 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=False,max_num_frames=1024 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_frame32_token256 \
    --output_path ./logs/
    # --verbosity=DEBUG