# python train.py --net spcnn_small --img-height 448 --img-width 448 --batch-size 24 --lr 3.0e-2 --dataset university1652 --gpus 0 --epochs 3,10 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.932953 top5:0.947218 top10:0.951498 mAP:0.914201
# python train.py --net spcnn_small --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-2 --dataset msmt17 --gpus 3 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.849129 top5:0.919033 top10:0.938502 mAP:0.648552
# python train.py --net spcnn_small --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-2 --dataset veri776 --gpus 4 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.969415 top5:0.981526 top10:0.987485 mAP:0.814716
# python train.py --net spcnn_small --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset deepfashion --gpus 5 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.936278 Recall@10:0.983472 Recall@20:0.989309 Recall@30:0.991419 Recall@40:0.992545 NMI:0.925350
# python train.py --net spcnn_small --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.822188 Recall@10:0.924333 NMI:0.916635


# python train.py --net inceptionnext_tiny --img-height 448 --img-width 448 --batch-size 24 --lr 3.0e-2 --dataset university1652 --gpus 0 --epochs 3,10 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.885877 top5:0.920114 top10:0.930100 mAP:0.856910
# python train.py --net inceptionnext_tiny --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.798782 top5:0.887555 top10:0.914744 mAP:0.552578
# python train.py --net inceptionnext_tiny --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-2 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.952324 top5:0.976758 top10:0.986293 mAP:0.768699
# python train.py --net inceptionnext_tiny --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset deepfashion --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.925447 Recall@10:0.981995 Recall@20:0.987903 Recall@30:0.990083 Recall@40:0.992193 NMI:0.915652
# python train.py --net spcnn_small --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.816700 Recall@10:0.922102 NMI:0.916312


# python train.py --net moganet_small --img-height 448 --img-width 448 --batch-size 24 --lr 3.0e-2 --dataset university1652 --gpus 0 --epochs 3,10 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.897290 top5:0.921541 top10:0.930100 mAP:0.863381
# python train.py --net moganet_small --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-2 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.763873 top5:0.864911 top10:0.898104 mAP:0.521204
# python train.py --net moganet_small --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-2 --dataset veri776 --gpus 1 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.927890 top5:0.961859 top10:0.973182 mAP:0.713426
# python train.py --net moganet_small --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset deepfashion --gpus 2 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.905824 Recall@10:0.976790 Recall@20:0.984456 Recall@30:0.987692 Recall@40:0.989591 NMI:0.908565
# python train.py --net moganet_small --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 3 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.768834 Recall@10:0.884913 NMI:0.899135


# python train.py --net unireplknet_t --img-height 448 --img-width 448 --batch-size 24 --lr 3.0e-2 --dataset university1652 --gpus 0 --epochs 3,10 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.910128 top5:0.932953 top10:0.938659 mAP:0.873970
# python train.py --net unireplknet_t --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-2 --dataset msmt17 --gpus 4 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.806330 top5:0.891414 top10:0.916717 mAP:0.568152
# python train.py --net unireplknet_t --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-2 --dataset veri776 --gpus 5 --epochs 5,75 --instance-num 4 --erasing 0.45 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# top1:0.936830 top5:0.970799 top10:0.979738 mAP:0.747974
# python train.py --net unireplknet_t --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset deepfashion --gpus 6 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.915248 Recall@10:0.979814 Recall@20:0.986637 Recall@30:0.989802 Recall@40:0.991701 NMI:0.913589
# python train.py --net unireplknet_t --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 7 --epochs 5,75 --instance-num 4 --erasing 0.15 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root ../datasets --feat-num 512
# Recall@1:0.808849 Recall@10:0.916680 NMI:0.911238

