# export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--n_class 7 \
--data_path "../../../data_2025/p2_data/" \
--model_path "./saved_models/" \
--log_path "./runs/" \
--task_name "fpn_deepglobe_global2local" \
--mode 2 \
--batch_size 32 \
--sub_batch_size 32 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_deepglobe_global.pth"