# Learning To Recognize Procedural Activities with Distant Supervision

This is an official pytorch implementation of [Learning To Recognize Procedural Activities with Distant Supervision](https://arxiv.org/abs/2201.10990). In this repository, we provide PyTorch code for training and testing as described in the paper. The proposed distant supervision framework achieves strong generalization performance on step classification, recognition of procedural activities, step forecasting (*New task proposed*) and egocentric activity classification.

If you find our repo useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{lin2022learning,
  title={Learning To Recognize Procedural Activities with Distant Supervision},
  author={Lin, Xudong and Petroni, Fabio and Bertasius, Gedas and Rohrbach, Marcus and Chang, Shih-Fu and Torresani, Lorenzo},
  journal={arXiv preprint arXiv:2201.10990},
  year={2022}
}
```
After cloining, please first unzip ./data_and_csv.zip.

# Distant Supervision

For flexible usage, we provide the extracted text embeddings of all the ASR sentences in HowTo100M and all the steps we used.  (link TBD)
We included the text of all the steps we used in [./data/step_label_text.json](./data/step_label_text.json). (Note that only 'headlines' are used.)



# Model Zoo

We provide [the TimeSformer model pretrained using our distant supervision](https://dl.fbaipublicfiles.com/video-distant-supervision/TimeSformer_divST_8x32_224_HowTo100M_pretrained.pth).

We also provide the models trained for recognition of procedural activities on COIN, step forecasting on COIN and egocentric activity classification on Epic-Kitchens-100.

|Long-term Model | Segment Model | Downstream Task | Acc (%) | Link |
| --- | --- | --- | --- | --- |
|Transformer w/ KB Transfer | TimeSformer | Recognition of Procedural Activities | 90.0 |TBD |
|Transformer w/ KB Transfer | TimeSformer | Step Forecasting | 39.4 |TBD |


| Segment Model | Downstream Task | Action (%) | Verb (%) | Noun (%) | Link |
| --- | --- | --- | --- | --- | --- |
| TimeSformer | Egocentric Activity Classification | 44.4 | 67.1 | 58.1 | TBD |


# Installation

First, create a conda virtual environment and activate it:
```
conda create -n distant python=3.7 -y
source activate distant
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- ffmpeg: `pip install ffmpeg-python`
- pandas: `pip install pandas`
- sentence-transformer: `pip install -U sentence-transformers`

Note that the part of this repo also requires ffmpeg installed in the system.

Lastly, include the absolute path of this directory in PYTHONPATH:
```
export PYTHONPATH="$PYTHONPATH:<Path_To_This_Directory>"
```
The alternative last step is to run the following command in this directory:
```
python setup.py build develop
```

# Usage

## Dataset Preparation

Please use the dataset preparation instructions provided in [DATASET.md](lib/datasets/DATASET.md).


## Pretraining with Distant Supervision

Training with the default distant supervision (Distribution matching Top-3) can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/HowTo100M/distribution_matching_top3.yaml \
  DATA.PATH_PREFIX path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_PREFIX`, or you can simply add

```
DATA:
  DATA.PATH_PREFIX: path_to_your_video_data
```

Note that the pretraining process needs ./data/step_to_task.pth to evaluate the zero-shot task classification performance on HowTo100M of the pretrained model. It is a matrix that projects a step to its cooresponding task.

To the yaml configs file, then you do not need to pass it to the command line every time.

We also provide configurations for other variants under [`configs/`](configs/).

## Using a Different Number of GPUs/Batch_Sizes

If you want to use a smaller number of GPUs, you need to modify .yaml configuration files in [`configs/`](configs/). Specifically, you need to modify the NUM_GPUS, TRAIN.BATCH_SIZE, TEST.BATCH_SIZE, DATA_LOADER.NUM_WORKERS entries in each configuration file. The BATCH_SIZE entry should be the same or larger as the NUM_GPUS entry. Note that gradient accumulation is integrated in the training function. If you want to change the global batch size for gradient descent, you need to update GLOBAL_BATCH_SIZE, which is 64 by default.



## Multinode Training

Distributed training is available via Slurm and submitit

```
pip install submitit
```

To reproduce our best pretrained model, use the following command:
```
python tools/submit.py --cfg configs/HowTo100M/distribution_matching_top3.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --name ${JOB_NAME} --use_volta32
```
after the model is trained, then run 
```
python tools/submit.py --cfg configs/HowTo100M/distribution_matching_top3_c30.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --name ${JOB_NAME} --use_volta32
```
We provide a few scripts for launching slurm jobs in [`slurm_scripts/`](slurm_scripts/).

## Evaluation

We provide configs and scripts for evaluation of the pretrained representation on COIN and Epic-Kitchen 100. Note that generally you need to set `DATA.PATH_PREFIX` and `TIMESFORMER.PRETRAINED_MODEL` to the correct location in the congiuration files before running scripts.

# Environment

The code was developed using python 3.7 on Ubuntu 20.04. For training, we used four GPU compute nodes each node containing 8 Tesla V100 GPUs (32 GPUs in total). Other platforms or GPU cards have not been fully tested.

# License

The majority of this work is licensed under CC-BY-NC-SA 3.0, however portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license.


# Contributing

We actively welcome your pull requests. Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.


# Acknowledgements

Thanks much to [Triantafyllos Afouras](afourast@fb.com) and [Mandy Toh](mandytoh@fb.com) for help on preration of the code release! 

This repo is built on top of [TimeSformer](https://github.com/facebookresearch/TimeSformer), [PySlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). We also borrow some implementation for Epic-Kitchen dataloader from [Motionformer](https://github.com/facebookresearch/Motionformer). We thank the authors for releasing their code. The Knowledge transfer retrieval implementation is inspired by [VX2TEXT](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Vx2Text_End-to-End_Learning_of_Video-Based_Text_Generation_From_Multimodal_Inputs_CVPR_2021_paper.pdf). If you use our model, please consider citing these works as well:
```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```
```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
```BibTeX
@inproceedings{patrick2021keeping,
   title={Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers}, 
   author={Mandela Patrick and Dylan Campbell and Yuki M. Asano and Ishan Misra Florian Metze and Christoph Feichtenhofer and Andrea Vedaldi and Jo\ão F. Henriques},
   year={2021},
   booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
}
```
```BibTeX
@inproceedings{lin2021vx2text,
  title={Vx2text: End-to-end learning of video-based text generation from multimodal inputs},
  author={Lin, Xudong and Bertasius, Gedas and Wang, Jue and Chang, Shih-Fu and Parikh, Devi and Torresani, Lorenzo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7005--7015},
  year={2021}
}
```