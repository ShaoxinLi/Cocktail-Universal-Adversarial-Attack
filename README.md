# Cocktail-Universal-Adversarial-Attack

Source code of **European Conference on Computer Vision (ECCV) 2024** paper "Cocktail Universal Adversarial Attack on Deep Neural Networks".

## Environment

- Python 3.8.10
- numpy==1.22.3
- scikit-learn==1.0.2
- scipy==1.8.0
- torch=1.11.0+cu113
- torchmetrics==0.6.0
- torchvision==0.13.1+cu113
- Pillow=9.1.0

You could use the following instruction to install all the requirements:

```
pip install -r requirements.txt
```

## Run UAP

As an example, to run the classic UAP attack against ResNet-50 on ImageNet dataset, you could use the following command:

```
python run_uap.py --run_uap.py --net_arch resnet50 --net_ckpt_path /path/to/model/checkpoint --data_root_dir /path/to/image/folder --dataset imagenet10
```

As an example, to run our proposed cocktail attack against ResNet-50 on ImageNet dataset, you could use the following command:

```
python run_our.py --net_arch resnet50 --assign_net_arch squeezenet --target_net_ckpt_path /path/to/target/model/checkpoint --data_root_dir /path/to/image/folder --dataset imagenet10 --k 5
```
