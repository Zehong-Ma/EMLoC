export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model longva \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=384,model_name=llava_qwen \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videomme_longva \
    --output_path ./logs/ 