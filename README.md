# MS-CapsNet
A multi-scale capsule network

## requirements
* mxnet 1.1.0
* pthon 3.6.0

## Table of context
1. [mutliscaleCapsule.py](./mutliscaleCapsule.py)
2. [msCapsuleLayer.py](./msCapsuleLayer.py)
3. [CapsuleLayer.py](./CapsuleLayer.py)
4. [embedding.py](./embedding.py)

## Results

| Methods | FashionMNIST(score) | CIFAR10(score) |
| :------: |:------:|:------:|
| CapsNet | 0.911 | 0.732 |
| MS-CapsNet | 0.922 | 0.751 |
| MS-CapsNet+Drop | 0.927 | 0.752 |
