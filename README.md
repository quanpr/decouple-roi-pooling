# decouple-roi-pooling
Decouple roi pooling for object detection. You might refer to [github](https://github.com/quanpr/decouple-roi-pooling)

## Preparation

First of all, clone the code
```
git clone https://github.com/quanpr/decouple-roi-pooling
```

### prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 (**now it does not support 0.4.1 or higher**)
* CUDA 8.0 or higher

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, create softlinks in the folder data/.

### Pretrained Model

We used Pytorch pretrained models in our experiments ResNet18. 

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train a decouple ROI pooling model with ResNet18 on pascal_voc, simply run:

```
	CUDA_VISIBLE_DEVICES=$GPU_ID python train_model.py --dataset pascal_voc_0712 --net res18 --bs 32 --nw 2 --lr 0.01 --lr_decay_step 10 --cuda --disp_interval 10 --epochs 30 --mGPUs --layers 18 --decouple 
```

## Test

We test our model thoroughly on dataset: pascal voc using ResNet18. Below are the results:

PASCAL VOC 07+12 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP
---------|--------|-----|--------|-----|-----|-------|--------|-----
[ResNet-18](https://drive.google.com/open?id=1UBwV8pLwBKNpRfJacJVYUl5qBcAuSo7k)     | 2 | 32 | 1e-2 | 10   | 23   |  0.67 hr | 10265MB   | 73.0
[ResNet-18](https://drive.google.com/open?id=1eQhU1Wf6JolPQ03uf_YVpvmf480YFHIF)     | 2 | 32 | 1e-2 | 10  | 27  |  0.69 hr | 17830MB   | 72.7