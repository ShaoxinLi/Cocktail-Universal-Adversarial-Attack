# Cocktail Universal Adversarial Attack on Deep Neural Networks

Official PyTorch implementation of our **ECCV 2024** paper.

Shaoxin Li, Xiaofeng Liao, Xin Che, Xintong Li, Yong Zhang, and Lingyang Chu

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08267.pdf) · [Supplementary material](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08267-supp.pdf) · [Citation](#citation)

Cocktail attack jointly trains a set of universal adversarial perturbations, or UAPs, and a selection network. Different UAPs target different groups of images. For each new image, the network selects one UAP to add, with no further training or fine-tuning.

## Selected results

Fooling ratio measures how often an attack changes the victim model's prediction. The table below reports percentages on the 50,000-image ImageNet validation set against ResNet-50. All methods use an L-infinity bound of 6/255. Cocktail attack uses five UAPs.

| Base method | Single UAP | Cocktail |
| :--- | ---: | ---: |
| UAT | 73.6 | 87.6 |
| DF-UAP | 72.7 | 87.0 |
| Cosine-UAP | 66.6 | 80.5 |
| NAG | 68.5 | 79.2 |

Cocktail results use the jointly trained variant, CK2. All values come from [Table 1 of the paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08267.pdf#page=10).

## Setup

Use Python 3.8.10 and an NVIDIA GPU for the CUDA 11.3 environment. [requirements.txt](requirements.txt) pins PyTorch 1.11.0 and torchvision 0.12.0. The plotting utilities also require LaTeX for the SciencePlots styles.

```bash
git clone https://github.com/ShaoxinLi/Cocktail-Universal-Adversarial-Attack.git
cd Cocktail-Universal-Adversarial-Attack
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

## Prepare ImageNet

Arrange the training and validation images in class folders. Use all 1,000 ImageNet classes with matching class-folder names in both splits.

```text
/home/share/Datasets/imagenet10/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Here, `imagenet10` is the code's name for the 1,000-class dataset. The commands below use `--n_samples 10000` to select ten training images per class and evaluate on the full validation set.

> The current loader hard-codes the validation root to `/home/share/Datasets/imagenet10` in [src/data/build.py](src/data/build.py). To use another location, replace that hard-coded string with `dataset_dir` and pass your dataset root through `--dataset_dir`. Changing `--data_root_dir` alone does not change the validation path.

## Train and evaluate

Run these commands from the repository root after preparing the data. They use torchvision's pretrained ResNet-50 weights, which download automatically on first use.

### Cocktail attack

Train five UAPs and a SqueezeNet selection network, then evaluate the attack:

```bash
python run_our.py \
  --net_arch resnet50 \
  --assign_net_arch squeezenet \
  --dataset imagenet10 \
  --data_root_dir /home/share/Datasets \
  --n_samples 10000 \
  --k 5 \
  --xi 6 \
  --batch_size 13 \
  --n_epochs 100 \
  --lr 0.001
```

### Single-UAP baseline

Train and evaluate one UAP with the same default loss, `neg_bounded_ce`:

```bash
python run_uap.py \
  --net_arch resnet50 \
  --dataset imagenet10 \
  --data_root_dir /home/share/Datasets \
  --n_samples 10000 \
  --validation_split -1 \
  --xi 6 \
  --batch_size 64 \
  --n_epochs 100 \
  --lr 0.001
```

`--xi` sets the perturbation bound in pixel units, so `--xi 6` means 6/255 on images scaled to [0, 1]. `--validation_split -1` keeps all selected training images for the baseline. See Section 4.1 of the paper for the full experimental settings.

Both scripts save checkpoints, training history, evaluation records, and plots under `archive/`. Use `--exp_root_dir` to change the output root. Run `python run_our.py --help` or `python run_uap.py --help` for all options. The repository also includes [run_nag.py](run_nag.py) for the NAG baseline and [run_classifier.py](run_classifier.py) for classifier training.

## Citation

If you use this work, please cite the paper. The BibTeX below follows [Springer's proceedings metadata](https://doi.org/10.1007/978-3-031-73650-6_23), which lists 2025 as the publication year for this ECCV 2024 paper.

```bibtex
@inproceedings{li2025cocktail,
  title     = {Cocktail Universal Adversarial Attack on Deep Neural Networks},
  author    = {Li, Shaoxin and Liao, Xiaofeng and Che, Xin and
               Li, Xintong and Zhang, Yong and Chu, Lingyang},
  booktitle = {Computer Vision -- ECCV 2024},
  pages     = {396--412},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-73650-6_23}
}
```
