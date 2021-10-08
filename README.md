# Semi-supervised-Federated-Learning-with-non-IID-Data-Algorithm-and-System-Design (HPCC'21)
Author: Zhe Zhang, Shiyao Ma, Zehui Xiong, Yi Wu, Qiang Yan, Xiaoke Xu, Dusit Niyato, Fellow, IEEE
## Abstract
In this paper, we design a Semi-supervised Federated Learning(SSFL) framework to slove classification problem.

----
## Dataset
We use the CIFAR-10 dataset including 56,000 training samples and 2,000 test samples as the validation dataset in our experiment. 
![](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

We also use the Fashion-MNIST dataset including 64,000 training samples and 2,000 test samples as the validation dataset.
![](https://codimd.xixiaoyao.cn/uploads/upload_9c41649d86cb07726c6b9d98dd6fbb8e.png)

Furthermore, we introduce Dirchlet distribution function to simulate the different non-IID level scenario in our experiment. We control Dirchlet distribution via modify parameters in  /modules/data_generator.py.
```python=
z = np.random.dirichlet((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), size=10)
```

----

## Framework
In this paper, we design a robust SSFL framework that uses the proposed FedMix algorithm to achieve high-precision semi-supervised learning.
![](https://codimd.xixiaoyao.cn/uploads/upload_625e1279e52f2a17729d28221c56e855.png)

And we make some improvements on the code of this [paper](https://arxiv.org/abs/2006.12097) and you also can cite this paper:
```
@inproceedings{jeong2020federated,
  title={Federated Semi-Supervised Learning with Inter-Client Consistency \& Disjoint Learning},
  author={Jeong, Wonyong and Yoon, Jaehong and Yang, Eunho and Hwang, Sung Ju},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```


----
## How to run
* step 1: Download the dataset and prepare the IID or Non-IID data via the following command lines:
```python=
python3 main.py -j data -t ls-biid-c10
python3 main.py -j data -t lc-bimb-c10
```
* step 2: If you want to run FedMix after generate the IID or Non-IID data, you can use the following command lines:
```python=
python main.py -g 0,1 -t ls-bimb-c10 -f 0.05
python main.py -g 0,1 -t ls-biid-c10 -f 0.05
```

----
## Experimental results
### Test accuracy curves on IID
![](https://codimd.xixiaoyao.cn/uploads/upload_543a9548e43354a18acfe34c66e1e967.png)


### Test accuracy curves on Non-IID
![](https://codimd.xixiaoyao.cn/uploads/upload_e5e2d0dfca6c70ae8b07e7bfc25023a6.png)

## Citation
```
Coming soon...
```
