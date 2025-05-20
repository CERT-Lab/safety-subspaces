lm_eval \
    --model hf \
    --model_args pretrained="give_model_path_here", dtype="bfloat16" \
    --tasks gsm8k \
    --num_fewshot 1 \
    --device cuda:0 \
    --batch_size 64
