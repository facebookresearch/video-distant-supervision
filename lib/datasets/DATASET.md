# Dataset Preparation

## General Instruction

We have provided data CSV files for pretraining on [full HowTo100M](https://github.com/facebookresearch/video-distant-supervision/data_csv/howto100m_full), pretraining on [a subset of HowTo100M](https://github.com/facebookresearch/video-distant-supervision/data_csv/howto100m_subset), [step classification on COIN](https://github.com/facebookresearch/video-distant-supervision/data_csv/coin_step), [recognition of procedural activities on COIN](https://github.com/facebookresearch/video-distant-supervision/data_csv/coin_task), [step forecasting on COIN](https://github.com/facebookresearch/video-distant-supervision/data_csv/coin_next).

Before you run experiments on any of them, you will need to set DATA.PATH_PREFIX in the configuration yaml file to the actual directory where the videos are stored. Note that due to posssibly different downloading stratergies, you may need to check the format of the video to be consistent between the data csv files and the actual videos you have.

## HowTo100M

Follow instructions from [dataset provider](https://www.di.ens.fr/willow/research/howto100m/) to download videos.
For the csv files of processed captions, we download from [MIL-NCE_HowTo100M](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip).

## COIN

Follow instructions from [dataset provider](https://coin-dataset.github.io/).

## Epic-Kitchens-100

Follow instructions from [dataset provider](https://github.com/epic-kitchens/epic-kitchens-100-annotations). Note that it also has a different dataloader structure, which directly loads the official anntoations under EPICKITCHENS.ANNOTATIONS_DIR of the configuration.
