# 👑 Input-Output-Conditioned Uncertainty Estimation (IO-CUE) 👑

**Paper:** https://arxiv.org/abs/2506.00918

A minimal implementation of the proposed IO-CUE framework, including all experiments shown in the submission.

## Getting the data

- This link provides the NYU Depth v2 and ApolloScape datasets as seen in Amini et al.: https://www.dropbox.com/s/qtab28cauzalqi7/depth_data.tar.gz?dl=1
- Unpack the data to a convenient path and make sure to update paths in dataloaders/simple_depth.py accordingly

## How to use the project

- Train a base model using the `train_base_model.py` script.
- Train a post-hoc learner using the `train_post_hoc_model.py` script and a config file. For example: `python train_post_hoc_model.py -yc configs/yaml_configs/edgy_depth_gaussian_io_cue.yaml -d 0`.
