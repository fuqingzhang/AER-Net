# [AER-Net: Adaptive Feature Enhancement and Hierarchical Refinement Network for Infrared Small Target Detection]
## Installation
```angular2html
pip install -U openmim
mim install mmcv-full==1.7.0
mim install mmdet==2.25.0
mim install mmsegmentation==0.28.0
```
You may also need to install other packages, if you encounter a package missing error, you just need to install it using the pip command.
## Dataset Preparation
### File Structure
```angular2html
|- datasets
   |- NUAA-SIRST
      |-trainval
        |-images
          |-Misc_1.png
          ......
        |-masks
          |-Misc_1.png
          ......
      |-test
        |-images
          |-Misc_50.png
          ......
        |-masks
          |-Misc_50.png
          ......
   |-NUDT-SIRST   
   |-IRSTD-1k
   |-BSIRST_v1

```
Please make sure that the path of your data set is consistent with the `data_root` in `configs/_base_/datasets/dataset_name.py`

## Training
### Single GPU Training

```
python train.py <CONFIG_FILE>
```

For example:

```
python train.py configs/network/nuaa.py
```

### Multi GPU Training

```nproc_per_node``` is the number of gpus you are using.

```
python -m torch.distributed.launch --nproc_per_node=[GPU_NUMS] train.py <CONFIG_FILE>
```

For example:

```
python -m torch.distributed.launch --nproc_per_node=4 train.py configs/network/nuaa.py
```

### Notes
* Be sure to set args.local_rank to 0 if using Multi-GPU training.

## Test

```
python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example:

```
python test.py configs/network/nuaa.py work_dirs/nuaa/20240828_214318/best.pth.tar
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

The default image save path is under <SEG_CHECKPOINT_FILE>. You can use `--work-dir` to specify the test log path, and the image save path is under this path by default. Of course, you can also use `--show-dir` to specify the image save path.


