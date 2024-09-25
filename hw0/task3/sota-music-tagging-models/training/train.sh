python -u main.py \
--num_workers 0 \
--dataset nsynth \
--model_type short \
--n_epochs 5 \
--batch_size 64 \
--use_tensorboard 1 \
--model_save_path ./../models/nsynth/no_to_db \
--log_step 20 \