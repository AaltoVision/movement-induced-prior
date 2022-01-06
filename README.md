# Movement-induced Priors for Deep Stereo 

### Dependencies
* pytorch
* pykitti
* torchvision

### Dataset
download [KITTI sequences](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

### Pretrained model from PSMNet
download pretrained KITTI 2015 from [PSMNet repo] (https://github.com/JiaRenChang/PSMNet)

### Run evaluation
```
python eval_kitti.py --gamma2 3.158501148223877 --ell 1.1633802652359009 --sigma2 1.2047102451324463 --ell2 0.6704858541488647
```
