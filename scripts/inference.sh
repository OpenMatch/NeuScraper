model_path=checkpoint-202312/checkpoint-202312.tar
data_path=data/test/TestNodes.json

CUDA_VISIBLE_DEVICES=0 python src/scraper/inference.py  \
	--model_path ${model_path} \
	--corpus_data_path ${data_path}