#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar10 \
--image_size 32 \
--model simclr \
--data_dir "./data/" \
--output_dir "outputs/cifar10_simclr/" \
--proj_layers 3 \
--backbone resnet50 \
--optimizer lars_simclr \
--weight_decay 0.000001 \
--momentum 0.9 \
--warmup_epochs 10 \
--warmup_lr 0 \
--base_lr 0.3 \
--final_lr 0 \
--num_epoch 200 \
--stop_at_epoch 200 \
--batch_size 256 \
--eval_after_train "--base_lr float(30)
                    --weight_decay float(0)
                    --momentum float(0.9)
                    --warmup_epochs int(0)
                    --batch_size int(256)
                    --num_epoch int(30)
                    --optimizer str('sgd')" \
--head_tail_accuracy 
# --hide_progress \

#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset cifar10 \
--image_size 32 \
--model byol \
--data_dir "./data/" \
--output_dir "outputs/cifar10_byol/" \
--proj_layers 3 \
--backbone resnet50 \
--optimizer lars_simclr \
--weight_decay 0.0000015 \
--momentum 0.9 \
--warmup_epochs 10 \
--warmup_lr 0 \
--base_lr 0.3 \
--final_lr 0 \
--num_epoch 200 \
--stop_at_epoch 200 \
--batch_size 256 \
--eval_after_train "--base_lr float(30)
                    --weight_decay float(0)
                    --momentum float(0.9)
                    --warmup_epochs int(0)
                    --batch_size int(256)
                    --num_epoch int(30)
                    --optimizer str('sgd')" \
--head_tail_accuracy 
# --hide_progress \