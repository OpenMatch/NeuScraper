parser_path=neuscraper-v1-clueweb/training_state_checkpoint.tar
path=commoncrawl/encoded/

CUDA_VISIBLE_DEVICES=1 python src/scraper/commoncrawl.py  \
	--model_path ${parser_path} \
	--corpus_root_path ${path}