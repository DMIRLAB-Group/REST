# REST

The PyTorch implementation of paper [REST: Debiased Social Recommendation via Reconstructing Exposure Strategies](xxx).

## Usage

### Download dataset

Download datasets: 
[Ciao](http://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip), 
[Epinions](http://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip), 
[Yelp](https://www.yelp.com/dataset/download)

Then unzip them into the directory `datasets` and run `preprocess_yelp.py`. 

```
└── datasets
    ├── Ciao
    │   ├── rating_with_timestamp.mat
    │   ├── trust.mat
    ├── Epinions
    │   ├── rating_with_timestamp.mat
    │   ├── trust.mat
    ├── Yelp
    │   ├── yelp_academic_dataset_review.json
    │   ├── yelp_academic_dataset_user.json
    │   ├── noiso_reid_u2uir.npz
    │   ├── ...
```

### Run

```bash
    python ./run_rate/run_rest_rate_ciao.py
```
During the training, we can obtain some logs and model-checkpoints in the directory `logs` and `saved_models`,

## Results
| Model | Ciao RMSE | Epinions RMSE | Yelp RMSE |
| :--:| :--: | :--: | :--: |
| PMF      | 1.1936±0.0019 | 1.2755±0.0022 | 1.2454±0.0011 |
| NeuMF    | 0.9828±0.0022 | 1.0838±0.0015 | 1.1958±0.0005 |
| MultiVAE | 1.1908±0.0014 | 1.2104±0.0039 | 1.2944±0.0020 |
| RecVAE   | 1.1787±0.0022 | 1.1946±0.0038 | 1.2385±0.0014 |
| CausE    | 1.0003±0.0013 | 1.0705±0.0013 | 1.2039±0.0015 |
| CVIB-MF  | 1.2001±0.0011 | 1.2477±0.0003 | 1.3189±0.0024 |
| CVIB-NCF | 1.0462±0.0013 | 1.2477±0.0003 | 1.3613±0.0043 |
| MACR-MF  | 1.1859±0.0030 | 1.2364±0.0031 | 1.2344±0.0004 |
| DecRS    | 0.9875±0.0033 | 1.0617±0.0033 | - |
| GraphRec | 0.9743±0.0021 | 1.0567±0.0019 | 1.1968±0.0017 |
| NGCF     | 1.0135±0.0010 | 1.1286±0.0017 | 1.2231±0.0017 |
| LightGCN | 1.1919±0.0014 | 1.2025±0.0005 | 1.2444±0.0019 |
| REST     | **0.9635±0.0009** | **1.0413±0.0007** | **1.1733±0.0006** |


Detailed results can be found in the [paper](xxx).

