model_path=neuscraper-v1-clueweb/training_state_checkpoint.tar
data_path=data/test/TestNodes.json

CUDA_VISIBLE_DEVICES=0 python src/scraper/inference.py  \
	--model_path ${model_path} \
	--data_path ${data_path}