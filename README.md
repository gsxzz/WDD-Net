# WDD-Net

Code for our manuscript submitted to The Visual Computer: "WDD-Net: A Small-object-sensitive Representation Learning Approach for Enhancing Weld Defect Detection".

## Quick Start

#### 1. Installation

```
pip install -r requirements.txt  
```

### 2. Dataset

The dataset can be downloaded from [weld defect dataset](https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection/releases/download/1.0/steel-tube-dataset-all.zip.). Put the dataset under `./datasets`.

#### 3. Train and test

```
#train
python train_dual.py
#test
python val_dual.py
```



