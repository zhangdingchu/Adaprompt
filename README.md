# Robust Test-Time Adaptation for Zero-Shot Prompt Tuning (ADAPROMPT)

This repository provides the official PyTorch implementation of our AAAI 2024 paper:

> Robust Test-Time Adaptation for Zero-Shot Prompt Tuning
> Authors: *Ding-Chu Zhang\*, Zhi Zhou\*, Yu-Feng Li*

## Environment

Install pip environment

`pip install -r requirements.txt`

Install conda environment

`conda install --yes --file requirements.txt`

## Datasets

Download the datasets [CIFAR10-C](https://zenodo.org/records/2535967), [CIFAR100-C](https://zenodo.org/records/3555552) and [TinyImageNet-C](https://zenodo.org/records/2536630).

Download cross-validation datasets by [TPT](https://github.com/azshue/TPT?tab=readme-ov-file).


## Run

1. Place the dataset in the `./MyDATA` folder

2. Run Adaprompt

    `python ./test.py ./MyDATA --test_sets CIFAR10_C -a ViT-B/16 -b 64 --gpu 0 --tpt --ctx_init a_photo_of_a --result-dir ./results/ours --method-config ./configs/methods/ours.yaml`

3. Run TPT

    `python ./test.py ./MyDATA --test_sets CIFAR10_C -a ViT-B/16 -b 64 --gpu 0 --tpt --ctx_init a_photo_of_a --result-dir ./results/tpt --method-config ./configs/methods/tpt.yaml `

4. Run Source

    `python ./test.py ./MyDATA --test_sets CIFAR10_C -a ViT-B/16 -b 64 --gpu 0 --tpt --ctx_init a_photo_of_a --result-dir ./results/source --method-config ./configs/methods/source.yaml `


5. The results will be printed and stored in `./results/.`

## Acknowledgment

We thank the authors for the following repositories for code reference: [TPT](https://github.com/azshue/TPT?tab=readme-ov-file).