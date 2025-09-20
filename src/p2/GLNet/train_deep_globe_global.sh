# export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--n_class 7 \
--data_path "../../../data_2025/p2_data/" \
--model_path "./saved_models/" \
--log_path "./runs/" \
--task_name "fpn_deepglobe_global" \
--mode 1 \
--batch_size 16 \
--sub_batch_size 16 \
--size_g 512 \
--size_p 512