# CAPTOR
This is the official implementation of our SIGIR'22 paper: **CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users**.

both PaddlePaddle and Pytorch versions are provided.
> PaddlePaddle: https://www.paddlepaddle.org.cn \
Pytorch: https://pytorch.org

If you use our codes in your research, please cite:
```
@inproceedings{xin2022captor,
  title={CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users},
  author={Xin, Haoran and Lu, Xinjiang and Zhu, Nengjun and Xu, Tong and Dou, Dejing and Xiong, Hui},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1174--1184},
  year={2022}
}
```
## Data
We have released the travel behavior dataset US which is generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) dataset.
You can run the model with the US out-of-town data provided in the "dataset" folder.

## Requirements
- python 3.x
- paddle 2.x / torch >= 1.7
- pgl / dgl>=0.6


## Run Our Model
Simply run the following command to train and evaluate:
```
cd ./PaddlePaddle
python run.py --ori_data {...} --dst_data {...} --trans_data {...} --pp_graph_path {...} ---save_path {...} --mode train --crf --memory --trans transd
```
