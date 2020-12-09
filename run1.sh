#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate simsiam
trap ctrl_c INT

function ctrl_c() {
        echo "** Trapped CTRL-C"
        exit
}
echo -n "Enter the card number: "
read COUNTRY
printf "You select card $COUNTRY\n"
case $COUNTRY in
    0)
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
        ;;
    1)
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
        ;;
    2)
        CUDA_VISIBLE_DEVICES=2 python main.py \
        --dataset cifar10 \
        --image_size 32 \
        --model simsiam \
        --data_dir "./data/" \
        --output_dir "./outputs/cifar10_simsiam_lars/" \
        --proj_layers 3 \
        --backbone resnet50 \
        --optimizer lars \
        --weight_decay 0.0005 \
        --momentum 0.9 \
        --warmup_epochs 10 \
        --warmup_lr 0 \
        --base_lr 0.03 \
        --final_lr 0 \
        --num_epochs 200 \
        --stop_at_epoch 200 \
        --batch_size 256 \
        --eval_after_train "--base_lr float(30)
                            --weight_decay float(0)
                            --momentum float(0.9)
                            --warmup_epochs int(0)
                            --batch_size int(256)
                            --num_epochs int(30)
                            --optimizer str('sgd')" \
        --head_tail_accuracy 
        # --hide_progress \
        ;;
    3)

        CUDA_VISIBLE_DEVICES=3 python main.py \
        --dataset cifar10 \
        --image_size 32 \
        --model simsiam \
        --data_dir "./data/" \
        --output_dir "outputs/cifar10_simsiam_sgd/" \
        --proj_layers 3 \
        --backbone resnet50 \
        --optimizer sgd \
        --weight_decay 0.0005 \
        --momentum 0.9 \
        --warmup_epochs 10 \
        --warmup_lr 0 \
        --base_lr 0.03 \
        --final_lr 0 \
        --num_epochs 200 \
        --stop_at_epoch 200 \
        --batch_size 256 \
        --eval_after_train "--base_lr float(30)
                            --weight_decay float(0)
                            --momentum float(0.9)
                            --warmup_epochs int(0)
                            --batch_size int(256)
                            --num_epochs int(30)
                            --optimizer str('sgd')" \
        --head_tail_accuracy 
        # --hide_progress \
        ;;
    *)
        echo -n "unknown"
        ;;
esac












