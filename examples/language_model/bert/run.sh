
# python create_pretraining_data.py \
#   --input_file=data/sample_text.txt \
#   --output_file=data/training_data.hdf5 \
#   --bert_model=bert-base-uncased \
#   --max_seq_length=128 \
#   --max_predictions_per_seq=20 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

# CUDA_VISIBLE_DEVICES=4 python3 run_pretrain.py \
# --model_type bert \
# --model_name_or_path bert-base-uncased \
# --input_dir data \
# --output_dir output \
# --max_steps 10

unset CUDA_VISIBLE_DEVICES
python3 -m paddle.distributed.launch --gpus "4" \
    --run_mode=collective \
    --log_dir log \
    run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 3 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 10 \
    --device gpu \
    --use_amp False \
    --seed 1000