# NeuScraper

Source code for our paper :  
***Cleaner Pretraining Corpus Curation with Neural Web Scraping***

If you find this work useful, please cite our paper  and give us a shining star 🌟



## Quick Start

**1️⃣ Clone from git**

```bash
git clone https://github.com/OpenMatch/NeuScraper
cd NeuScraper
```

**2️⃣ Data**

ClueWeb22 is the newest in the Lemur Project's ClueWeb line of datasets that support research on information retrieval, natural language processing and related human language technologies. 

The ClueWeb22 datasets are distributed by Carnegie Mellon University for research purposes only. A dataset may be obtained by signing a data license agreement with Carnegie Mellon University, and paying a fee that covers the cost of distributing the dataset. For details on how to get it, please click the following link:

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

```
We'll release checkpoints on 🤗HuggingFace this week
```

**2️⃣ Preprocess the test data, we use the** `en0001-01` **as our test set.**

```bash
python src/build_test.py \
		--path /path/to/clueweb22
```

**3️⃣ Scraping with NeuScraper**

```bash
bash scripts/inference.sh
```

**4️⃣ Test on** `en0001-01`

```bash
python src/eval/run_eval.py
```



## Main Result 

The results are shown as follows.

| **Method**     | **Acc.**  | **Prec.** | **Rec.**  | **F1**    |
| -------------- | --------- | --------- | --------- | --------- |
| htmlparser     | 40.94     | 40.92     | 98.95     | 57.90     |
| bs4            | 41.07     | 41.05     | **99.94** | 58.20     |
| html2text      | 40.09     | 39.40     | 85.40     | 53.92     |
| boilerpipe     | 66.28     | 66.89     | 35.52     | 46.40     |
| jusText        | 62.67     | 72.49     | 27.06     | 39.41     |
| lxml           | 65.45     | 61.54     | 37.82     | 46.84     |
| inscriptis     | 45.06     | 42.53     | 96.43     | 59.03     |
| readability    | 68.26     | 72.08     | 37.01     | 48.91     |
| trafilatura    | 70.57     | 66.60     | 56.77     | 61.30     |
| **NeuScraper** | **86.66** | **81.15** | 88.30     | **84.58** |



## Train NeuScraper from scratch 

***Note:** Training NeuScraper from scratch needs to be done on a server equipped with 8 NVIDIA A100-40G GPUs and SSDs*

1️⃣ **We need to preprocess the pages in Clueweb22:**

```bash
python src/build_train.py \
		--path /path/to/clueweb22
```

This command will place the processed data in `data/train`.  
It need to slice some of them up and put them in `data/val`.

2️⃣ **Run the following script to start training**

```bash
bash scripts/train.sh
```

The training process will run for 30 epochs and take about 40 hours. 



## CommonCrawl Support

We will add support for CommonCrwal in two months.



## Contact Us

If you have questions, suggestions, and bug reports, please send a email to us, we will try our best to help you. 

```bash
xuzhipeng@stumail.neu.edu.cn  
```