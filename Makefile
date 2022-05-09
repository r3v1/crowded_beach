.ONESHELL:
init:
	pip install -U -r requirements.txt

.ONESHELL:
download_checkpoints:
	mkdir -p coco_det_finetune/resnet_1024x1024
	gsutil -m cp "gs://pix2seq/coco_det_finetune/resnet_1024x1024/checkpoint" "gs://pix2seq/coco_det_finetune/resnet_1024x1024/ckpt-93324.data-00000-of-00001" "gs://pix2seq/coco_det_finetune/resnet_1024x1024/ckpt-93324.index" "gs://pix2seq/coco_det_finetune/resnet_1024x1024/config.json" coco_det_finetune/resnet_1024x1024
