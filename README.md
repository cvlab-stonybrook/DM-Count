# DM-Count

Official Pytorch implementation of the paper Distribution Matching for Crowd Counting (NeurIPS, spotlight).

[Arxiv](https://arxiv.org/pdf/2009.13077.pdf) | [NeurIPS Processings](https://proceedings.neurips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html) | :rocket: [Web Demo and Docker image on Replicate](https://replicate.com/cvlab-stonybrook/crowd-counting)

We propose to use Distribution Matching for crowd COUNTing (DM-Count). In DM-Count, we use Optimal Transport (OT) to measure the similarity between the normalized predicted density map and the normalized ground truth density map. To stabilize OT computation, we include a Total Variation loss in our model. We show that the generalization error bound of DM-Count is tighter than that of the Gaussian smoothed methods. Empirically, our method outperforms the state-of-the-art methods by a large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50.


## Prerequisites

Python 3.x

Pytorch >= 1.2

For other libraries, check requirements.txt.

## Getting Started
1. Dataset download

+ QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)

+ NWPU can be downloaded [here](https://www.crowdbenchmark.com/nwpucrowd.html)

+ Shanghai Tech Part A and Part B can be downloaded [here](https://www.kaggle.com/tthien/shanghaitech)

2. Data preprocess

Due to large sizes of images in QNRF and NWPU datasets, we preprocess these two datasets.

```
python preprocess_dataset.py --dataset <dataset name: qnrf or nwpu> --input-dataset-path <original data directory> --output-dataset-path <processed data directory> 
```
    
3. Training

```
python train.py --dataset <dataset name: qnrf, sha, shb or nwpu> --data-dir <path to dataset> --device <gpu device id>
```

4. Test

```
python test.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset> --dataset <dataset name: qnrf, sha, shb or nwpu>
```

## Pretrained models

Pretrained models on UCF-QNRF, NWPU, Shanghaitech part A and B can be found [Google Drive](https://drive.google.com/drive/folders/10U7F4iW_aPICM5-qJq21SXLLkzlum9tX?usp=sharing). You could download them and put them in in pretrained_models folder.


## Other resources

+ Web Demo

**6.26.2022** 

### Replicate 

:rocket: A Web Demo and Docker image can be found on Replicate [here](https://replicate.com/google-research/3d-moments). You can upload an image and choose between the 4 different models to run inference on. 

---

### Gradio

A web interface to can be found [here](https://gradio.app/g/dm-count). 

![demo](https://i.ibb.co/m65HpCJ/dm-count.gif)

Feel free to upload a image and try out the demo on a web browser. It is developed by [Ali Abdalla](twitter.com/si3luwa) from [Gradio](https://github.com/gradio-app/gradio). Gradio is an open source library, which helps to create interfaces to make models more accessible. Thanks Ali and Gradio! 

To launch a Gradio interface, run 

```
 python demo.py
```

+ Kaggle Notebook

A [Kaggle Notebook](https://www.kaggle.com/selmanzleyen/dmcount-shb) is developed by [Selman Ozleyen](https://github.com/SelmanOzleyen/DM-Count). Thanks Selman!


## References
If you find this work or code useful, please cite:

```
@inproceedings{wang2020DMCount,
  title={Distribution Matching for Crowd Counting},
  author={Boyu Wang and Huidong Liu and Dimitris Samaras and Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020},
}
```
