#!bin/bash
version=512 ##1024, 512, 256
seed=111
name=dynamicrafter_512_seed${seed}


ckpt="/workspace/DynamiCrafter/epoch=2-step=13581-weight.ckpt.ckpt"
# ckpt="/workspace/DynamiCrafter/epoch=0-step=4527-weight.ckpt.ckpt"
# ckpt="/workspace/epoch=4-step=22635-weight.ckpt"
# ckpt="/workspace/DynamiCrafter/model.ckpt"
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/512/
res_dir="results"


H=320
FS=15 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)

 
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae

## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop