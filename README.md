# DRTENet: Dual-Focus Residual Tensor Enhancement Network for Infrared Small Target Detection

![frame](frame.png)
## Abstract
Infrared small target detection remains a challenging task due to the inherent limitations of complex backgrounds and the absence of distinct target features. To address these challenges, we propose a Dual-Focus Residual Tensor Enhancement Network (DRTENet) which integrates edge enhancement and noise suppression. We propose a residual tensor weighting module (RTWM) that computes the local structure tensors to yield edge-saliency maps and integrates a residual learning strategy to preserve target contours and suppress background clutters. Based on the RTWM, we construct a Gaussian pyramid encoder to excavate multi-scale edge features and smooth point noise. Furthermore, we propose a dual-focus optimization module that designs a two-branch structure to enhance small targets powered by local semantic content, while simultaneously reducing background noise through global contextual information. Extensive experiments on several public infrared datasets demonstrate the effectiveness of DRTENet, achieving state-of-the-art detection performance.

## Prerequisite
- python == 3.8
- pytorch == 1.10
- CUDA == 11.1
- mmcv-full == 1.7.0
- mmdet == 2.25.0
- mmsegmentation == 0.28.0

## Datasets
- You can download them directly from the website: [NUAA](https://www.scidb.cn/en/detail?dataSetId=720626420933459968), [IRSDT](https://www.scidb.cn/en/detail?dataSetId=de971a1898774dc5921b68793817916e&dataSetType=journal), [NUDT](https://xzbai.buaa.edu.cn/datasets.html).

## Usage
### Train
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
### Test
- Usually model_best.pth is not necessarily the best model. The best model may have a lower val_loss or a higher AP50 during verification.
```
CUDA_VISIBLE_DEVICES=0 python vid_map_coco.py
```
### Visulization
```
python vid_predict.py
```

## Citation
If you find this repo useful, please cite our paper.
```


```
