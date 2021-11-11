# GreaseLM: Graph REASoning Enhanced Language Models

This repo provides the source code & data of our paper "GreaseLM: Graph REASoning Enhanced Language Models".

<p align="center">
  <img src="./figs/greaselm.png" width="600" title="GreaseLM model architecture" alt="">
</p>

## Usage
### 1. Dependencies

- [Python](<https://www.python.org/>) == 3.8
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.8.0
- [transformers](<https://github.com/huggingface/transformers/tree/v3.4.0>) == 3.4.0
- [torch-geometric](https://pytorch-geometric.readthedocs.io/) == 1.7.0

Run the following commands to create a conda environment (assuming CUDA 10.1):
```bash
conda create -y -n greaselm python=3.8
conda activate greaselm
pip install numpy==1.18.3 tqdm
pip install torch==1.8.0+cu101 torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0 nltk spacy
pip install wandb
conda install -y -c conda-forge tensorboardx
conda install -y -c conda-forge tensorboard

# for torch-geometric
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
```


### 2. Download data

Download all the raw data -- ConceptNet, CommonsenseQA, OpenBookQA -- by
```
./download_raw_data.sh
```

You can preprocess the raw data by running
```
CUDA_VISIBLE_DEVICES=0 python preprocess.py -p <num_processes>
```
You can specify the GPU you want to use in the beginning of the command `CUDA_VISIBLE_DEVICES=...`. The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair

**TL;DR**. The preprocessing may take long; for your convenience, you can download all the processed data [here](https://drive.google.com/drive/folders/1T6B4nou5P3u-6jr0z6e3IkitO8fNVM6f?usp=sharing) into the top-level directory of this repo and run
```
unzip data_preprocessed.zip
```

**Add MedQA-USMLE**. Besides the commonsense QA datasets (*CommonsenseQA*, *OpenBookQA*) with the ConceptNet knowledge graph, we added a biomedical QA dataset ([*MedQA-USMLE*](https://github.com/jind11/MedQA)) with a biomedical knowledge graph based on Disease Database and DrugBank. You can download all the data for this from [[here]](https://drive.google.com/file/d/1EqbiNt2ACXVrc9gmoXnzTEo9GJTe9Uor/view?usp=sharing). Unzip it and put the `medqa_usmle` and `ddb` folders inside the `data/` directory.


The resulting file structure should look like this:

```plain
.
├── README.md
└── data/
    ├── cpnet/                 (preprocessed ConceptNet)
    └── csqa/
        ├── train_rand_split.jsonl
        ├── dev_rand_split.jsonl
        ├── test_rand_split_no_answers.jsonl
        ├── statement/             (converted statements)
        ├── grounded/              (grounded entities)
        ├── graphs/                (extracted subgraphs)
        ├── ...
```

### 3. Training GreaseLM
To train GreaseLM on CommonsenseQA, run
```
CUDA_VISIBLE_DEVICES=0 ./run_greaselm.sh csqa --data_dir data/
```
You can specify up to 2 GPUs you want to use in the beginning of the command `CUDA_VISIBLE_DEVICES=...`.

Similarly, to train GreaseLM on OpenbookQA, run
```
CUDA_VISIBLE_DEVICES=0 ./run_greaselm.sh obqa --data_dir data/
```

To train GreaseLM on MedQA-USMLE, run
```
CUDA_VISIBLE_DEVICES=0 ./run_greaselm__medqa_usmle.sh
```

### 4. Pretrained model checkpoints
You can download a pretrained GreaseLM model on CommonsenseQA [here](https://drive.google.com/file/d/1QPwLZFA6AQ-pFfDR6TWLdBAvm3c_HOUr/view?usp=sharing), which achieves an IH-dev acc. of `79.0` and an IH-test acc. of `74.0`.

You can also download a pretrained GreaseLM model on OpenbookQA [here](https://drive.google.com/file/d/1-QqyiQuU9xlN20vwfIaqYQ_uJMP8d7Pv/view?usp=sharing), which achieves an test acc. of `84.8`.

You can also download a pretrained GreaseLM model on MedQA-USMLE [here](https://drive.google.com/file/d/1x5nZEprV0Ht8IWViyz3d07uGLXtNjUN1/view?usp=sharing), which achieves an test acc. of `38.5`.

### 5. Evaluating a pretrained model checkpoint
To evaluate a pretrained GreaseLM model checkpoint on CommonsenseQA, run
```
CUDA_VISIBLE_DEVICES=0 ./eval_greaselm.sh csqa --data_dir data/ --load_model_path /path/to/checkpoint
```
Again you can specify up to 2 GPUs you want to use in the beginning of the command `CUDA_VISIBLE_DEVICES=...`.

SimilarlyTo evaluate a pretrained GreaseLM model checkpoint on OpenbookQA, run
```
CUDA_VISIBLE_DEVICES=0 ./eval_greaselm.sh obqa --data_dir data/ --load_model_path /path/to/checkpoint
```

### 6. Use your own dataset
- Convert your dataset to  `{train,dev,test}.statement.jsonl`  in .jsonl format (see `data/csqa/statement/train.statement.jsonl`)
- Create a directory in `data/{yourdataset}/` to store the .jsonl files
- Modify `preprocess.py` and perform subgraph extraction for your data
- Modify `utils/parser_utils.py` to support your own dataset

## Acknowledgment
This repo is built upon the following work:
```
QA-GNN: Question Answering using Language Models and Knowledge Graphs
https://github.com/michiyasunaga/qagnn
```
Many thanks to the authors and developers!
