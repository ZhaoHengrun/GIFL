# [AAAI 2024] Large Occluded Human Image Completion via Image-Prior Cooperating

## Requirements

```
pytorch
torchvision
numpy
random
glob
einops
timm
pyiqa
opencv
```

## Prepare dataset
Download AHP dataset ```https://sydney0zq.github.io/ahp/```  
Download object mask ```https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/EUcTzYoX5LhFkBCjG62E5wwBzru1eM4PhwmRNGmo08pH5Q?download=1```  
Run ```python crop_AHP_dataset.py``` to crop the image  
Run ```python dataset/crop_obj_mask.py``` to crop the object mask for test  
Run ```python dataset/generate_mask_for_test.py``` to generate the center mask for test  
Move all images with the word ```test``` in the file name to ```dataset/test/```  
## Train
1. Pre-training  ```python train_pretrain.py```
2. Train the coarse network ```python train_corse.py```
3. Train the refinement network ```python train_refinement.py```
4. Train the segmentation network ```python train_unet_seg.py```
## Test
1. We provided a trained weight:https://drive.google.com/file/d/1cns3WYNf0lKQytzDOi2BlQF8P7WdhCsb/view?usp=sharing
2. Test ```python test.py```
