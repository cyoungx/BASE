file_path='./results'
experiment_id='XXXXXX'
timestamp=$(date +"%Y%m%d%H%M%S")

CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-set/ctrgcn_default.yaml \
                                            --work-dir ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default \
                                            --device 0 \
                                            --phase train \
                                            --file_path $file_path \
                                            --experiment_id $experiment_id \
                                            --timestamp $timestamp \
                                            --step 25 45 60 \
                                            --num_epoch 60 \
                                            --save_epoch 30 \
                                            --eval_interval 10 \
                                            --run 2 \
                                            --num_class 30 \
                                            --dim 256 \
                                            --flag_loss_edl \
                                            --flag_loss_aes \
                                            --flag_semlp \
                                            --flag_text_proj \
                                            --text_gpt4_path data/language/ntu120_des_gpt4_embeddings.npy \
                                            --flag_module_cmi \
                                            --flag_cmi_lhsg \
                                            --flag_loss_enh \
                                            --flag_cmi_bpom \
                                            --flag_loss_bcl \

CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/nturgbd120-cross-set/ctrgcn_default.yaml \
                                            --device 0 \
                                            --phase test \
                                            --file_path $file_path \
                                            --experiment_id $experiment_id \
                                            --timestamp $timestamp \
                                            --weights ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-60*.pt \
                                            --weights_velocity ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-velocity60*.pt \
                                            --weights_bone ${file_path}/${experiment_id}_${timestamp}/work_dir/ctrgcn/ntu120/cset/default/runs-bone60*.pt \
                                            --num_class 30 \
                                            --run 2 \
                                            --flag_semlp \
