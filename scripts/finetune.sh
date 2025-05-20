CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model "path_to_aligned_model" \
    --data_mode pure \
    --batch_size 1 \
    --epochs 1 \
    --lr 1e-5

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model "path_to_aligned_model" \
    --data_mode contaminated \
    --batch_size 1 \
    --epochs 1 \
    --lr 1e-5

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model "path_to_aligned_model" \
    --data_mode harmful \
    --batch_size 1 \
    --epochs 1 \
    --lr 1e-5



