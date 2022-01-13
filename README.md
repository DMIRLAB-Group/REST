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
| Model | Ciao MAE/RMSE | Epinions MAE/RMSE | Yelp MAE/RMSE |
| :--:| :--: | :--: | :--: |
| PMF      | 0.9539±0.0040 / 1.1936±0.0019 | 1.0767±0.0035 / 1.2755±0.0022 | 0.9896±0.0023 / 1.2454±0.0011 |
| NeuMF    | 0.7770±0.0077 / 0.9828±0.0022 | 0.8457±0.0053 / 1.0838±0.0015 | 0.9575±0.0081 / 1.1958±0.0005 |
| MultiVAE | 0.9254±0.0025 / 1.1908±0.0014 | 0.9707±0.0104 / 1.2104±0.0039 | 0.9957±0.0031 / 1.2944±0.0020 |
| RecVAE   | 0.9449±0.0014 / 1.1787±0.0022 | 0.9614±0.0087 / 1.1946±0.0038 | 0.9944±0.0020 / 1.2385±0.0014 |
| CausE    | 0.7943±0.0014 / 1.0003±0.0013 | 0.8553±0.0019 / 1.0705±0.0013 | 0.9400±0.0031 / 1.2039±0.0015 |
| CVIB-MF  | 0.9091±0.0016 / 1.2001±0.0011 | 0.9499±0.0031 / 1.2477±0.0003 | 0.9919±0.0122 / 1.3189±0.0024 |
| CVIB-NCF | 0.7394±0.0027 / 1.0462±0.0013 | 0.8311±0.0128 / 1.2477±0.0003 | 0.9801±0.0011 / 1.3613±0.0043 |
| MACR-MF  | 0.9446±0.0051 / 1.1859±0.0030 | 0.9784±0.0092 / 1.2364±0.0031 | 0.9923±0.0004 / 1.2344±0.0004 |
| DecRS    | 0.7576±0.0038 / 0.9875±0.0033 | 0.8242±0.0043 / 1.0617±0.0033 | - |
| GraphRec | 0.7585±0.0051 / 0.9743±0.0021 | 0.8283±0.0019 / 1.0567±0.0019 | 0.9525±0.0035 / 1.1968±0.0017 |
| NGCF     | 0.8061±0.0023 / 1.0135±0.0010 | 0.9348±0.0023 / 1.1286±0.0017 | 0.9396±0.0023 / 1.2231±0.0017 |
| LightGCN | 0.9373±0.0051 / 1.1919±0.0014 | 0.9584±0.0011 / 1.2025±0.0005 | 1.0015±0.0024 / 1.2444±0.0019 |
| REST     | **0.7320±0.0117 / 0.9635±0.0009** | **0.8013±0.0045 / 1.0413±0.0007** | **0.9158±0.0054 / 1.1733±0.0006** |


Detailed results can be found in the [paper](xxx).

