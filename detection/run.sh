# python tools/analysis_tools/get_flops.py configs/retinanet/retinanet_spcnn_small_fpn_fp16_1x_coco.py --shape 1280 800
# python tools/analysis_tools/get_flops.py configs/retinanet/retinanet_spcnn_base_fpn_fp16_1x_coco.py --shape 1280 800
# python tools/analysis_tools/get_flops.py configs/mask_rcnn/mask-rcnn_spcnn_small_fpn_fp16_1x_coco.py --shape 1280 800
# python tools/analysis_tools/get_flops.py configs/mask_rcnn/mask-rcnn_spcnn_base_fpn_fp16_1x_coco.py --shape 1280 800

# bash ./tools/dist_train.sh configs/retinanet/retinanet_spcnn_small_fpn_fp16_1x_coco.py 8
# bash ./tools/dist_train.sh configs/retinanet/retinanet_spcnn_base_fpn_fp16_1x_coco.py 8
# bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_spcnn_small_fpn_fp16_1x_coco.py 8
# bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_spcnn_base_fpn_fp16_1x_coco.py 8