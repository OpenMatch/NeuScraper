# NeuScraper

Source code for our paper :  
***[Cleaner Pretraining Corpus Curation with Neural Web Scraping](https://arxiv.org/abs/2402.14652)***  
***This is a work in progress, so please wait for our incremental improvements 😊***

If you find this work useful, please cite our paper  and give us a shining star 🌟



## Quick Start

**1️⃣ Clone from git**

```bash
git clone https://github.com/OpenMatch/NeuScraper
cd NeuScraper
```

**2️⃣ Data**

ClueWeb22 is the newest in the Lemur Project's ClueWeb line of datasets that support research on information retrieval, natural language processing and related human language technologies. 

The ClueWeb22 datasets are distributed by Carnegie Mellon University for research purposes only. A dataset may be obtained by signing a data license agreement with Carnegie Mellon University. For details on how to get it, please click the following link:

```bash
https://www.lemurproject.org/clueweb22/obtain.php
```

**3️⃣ Environment**

Install the `torch` first :

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other packages :

```bash
pip install -r requirements.txt
```



## Reproduction

**1️⃣ Download checkpoint for NeuScraper**

```bash
git lfs install
git clone https://huggingface.co/OpenMatch/neuscraper-v1-clueweb
```

**2️⃣ Preprocess the test data, we use the** `en0001-01` **as our test set.**

```bash
python src/build_test.py --path /path/to/clueweb22
```

**3️⃣ Scraping with NeuScraper**

```bash
bash scripts/inference.sh
```

**4️⃣ Test on** `en0001-01`

```bash
python src/eval/run_eval.py
```



## Train NeuScraper from scratch 

***Note:** Training NeuScraper from scratch needs to be done on a server equipped with 8 NVIDIA A100-40G GPUs and SSDs*

1️⃣ **We need to preprocess the pages in Clueweb22:**

```bash
python src/build_train.py --path /path/to/clueweb22
```

This command will place the processed data in `data/train`.  
It need to slice some of them up and put them in `data/val`.

2️⃣ **Run the following script to start training**

```bash
bash scripts/train.sh
```

The training process will run for 30 epochs and take about 40 hours. 



## CommonCrawl Support (Beta)

***Note:** CommonCrawl support is still a beta version and it still needs to be more efficient. Besides, neuscraper-v1 only focuses on scraping in English web pages. We are still trying our best to develop the new version, and have opened the MIT license. So if you want to develop a better version, do what you want!*

1️⃣ **Preprocess the pages in CommonCrawl**

```bash
python src/warc/build.py --path /path/to/commoncrawl/warc
```

2️⃣ **Scraping by NeuScraper**

```bash
python scripts/commoncrawl.sh
```

3️⃣ **Get Text**

```bash
python src/warc/get_text.py
```



## Citation

```
@article{xu2024cleaner,
  title={Cleaner Pretraining Corpus Curation with Neural Web Scraping},
  author={Xu, Zhipeng and Liu, Zhenghao and Yan, Yukun and Liu, Zhiyuan and Xiong, Chenyan and Yu, Ge},
  journal={arXiv preprint arXiv:2402.14652},
  year={2024}
}
```



## Contact Us

If you have questions, suggestions, and bug reports, please send a email to us, we will try our best to help you. 

```bash
xuzhipeng@stumail.neu.edu.cn  
```
