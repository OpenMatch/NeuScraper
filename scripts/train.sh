CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 19680 src/scraper/trainer.py  \
	--output_dir checkpoints \
	--data_dir data