## Multi-Source Decentralized Transfer for Privacy-Preserving BCIs

This repository contains codes of our paper https://ieeexplore.ieee.org/document/9894428



## Datasets
The MI datasets can be downloaded and processed with MOABB in http://moabb.neurotechx.com/docs/datasets.html




## Prerequisites:

- python == 3.7.6

- pyriemann == 0.2.6

- PyTorch == 1.8.0

- mne == 0.20.7

- numpy, scipy, sklearn

  

## Running the code


Code files introduction:

**utils/** -- necessary function files

**source_train_multi_mi.py** -- demo file, source models pre-training.

**target_adapt_msdt_mi.py** -- demo file,  gray box MSDT.

**target_adapt_msdt_kd.py** -- demo file, black box MSDT.



## Notes

The codes are only for reference. In the early version, in the model pre-training stage, we set the learning rate of the feature extractor to 1/10 of the feature extractor, which has been revised as the same learning rate. The cross-subject classification results with this version are similar to the paper.

