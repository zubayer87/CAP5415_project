# Camera Style Adaptation for Person Re-identification
================================================================

Source: https://github.com/zhunzhong07/CamStyle

** This source code has been modified to incorporate the mask generated by PSPNet during the Cycle-GAN training.

### Preparation

#### Requirements: Python=3.6 and Pytorch>=0.3.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset
   
   - Market-1501   [[BaiduYun]](https://pan.baidu.com/s/1ntIi2Op) [[GoogleDriver]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
   
   - DukeMTMC-reID   [[BaiduYun]](https://pan.baidu.com/share/init?surl=kUD80xp) (password: chu1) [[GoogleDriver]](https://drive.google.com/file/d/0B0VOCNYh8HeRdnBPa2ZWaVBYSVk/view)
   
   - Move them to 'CamStyle/data/market (or duke)'
   

3. Download CamStyle Images
   
   - Market-1501-Camstyle [[GoogleDriver]](https://drive.google.com/open?id=1z9bc-I23OyLCZ2eTms2NTWSq4gePp2fr)
   
   - DukeMTMC-reID-CamStyle  [[GoogleDriver]](https://drive.google.com/open?id=1QX3K_RK1wBPPLQRYRyvG0BIf-bzsUKbt)
   
   - Move them to 'CamStyle/data/market (or duke)/bounding_box_train_camstyle'


### CamStyle Generation
You can generate CamStyle imgaes with [CycleGAN-for-CamStyle](https://github.com/zhunzhong07/CamStyle/tree/master/CycleGAN-for-CamStyle)


### Training and test re-ID model

1. IDE
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide
  ```
2. IDE + CamStyle
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle --camstyle 46
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle --camstyle 46
  ```
  
3. IDE + CamStyle + Random Erasing[4]
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle-re --camstyle 46 --re 0.5
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle-re --camstyle 46 --re 0.5
  ```

4. IDE + CamStyle + Random Erasing[4] + re-ranking[3]
  ```Shell
  # For Market-1501
  python main.py -d market --logs-dir logs/market-ide-camstyle-re --camstyle 46 --re 0.5 --rerank
  # For Duke
  python main.py -d duke --logs-dir logs/duke-ide--camstyle-re --camstyle 46 --re 0.5 --rerank
  ```
  


### References

- [1] Our code is conducted based on [open-reid](https://github.com/Cysu/open-reid)

- [2] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017

- [3] Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017.

- [4] Random Erasing Data Augmentation. Arxiv 2017.




