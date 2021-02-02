# Scene Graph Benchmark in PyTorch 1.4

**This project is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)**

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models using PyTorch 1.0.

![alt text](demo/R152FPN_demo.png "from https://storage.googleapis.com/openimages/web/index.html")


## Highlights
- **Upgrad to pytorch 1.4**
- **Multi-GPU training and inference**
- **Mixed precision training:** trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores).
- **Batched inference:** can perform inference using multiple images per batch per GPU.
- **Fast and flexible tsv dataset format**
- **Remove FasterRCNN detector dependency:** during relation head training, can plugin bounding boxes from any detector.
- Provides pre-trained models for different scene graph detection algorithms ([IMP](https://arxiv.org/pdf/1701.02426.pdf), [MSDN](http://cvboy.com/publication/iccv2017_msdn/), [GRCNN](https://arxiv.org/pdf/1808.00191.pdf), [Neural Motif](https://arxiv.org/pdf/1711.06640.pdf), [RelDN](https://arxiv.org/pdf/1903.02728.pdf)).
- Provides bounding box level feature extraction functionality
- Provides large detector backbones (ResNxt152)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Model Zoo and Baselines

Pre-trained models, baselines and comparison with Detectron and mmdetection
can be found in [SCENE_GRAPH_MODEL_ZOO.md](SCENE_GRAPH_MODEL_ZOO.md)


## Visualization and Demo
We provide a helper class to simplify writing inference pipelines using pre-trained models (Currently only support objects and attributes).
Here is how we would do it. Run the following commands:
```bash
python tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ../maskrcnn-benchmark-1/datasets1/imgs/woman_fish.jpg --save_file output/woman_fish_x152c4.obj.jpg MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "../maskrcnn-benchmark-1/datasets1" TEST.IGNORE_BOX_REGRESSION False

python tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file ../maskrcnn-benchmark-1/datasets1/imgs/woman_fish.jpg --save_file output/woman_fish_x152c4.attr.jpg --visualize_attr MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "../maskrcnn-benchmark-1/datasets1" TEST.IGNORE_BOX_REGRESSION False
```

## Perform training

For the following examples to work, you need to first install this repo.

You will also need to download the dataset.
We recommend to symlink the path to the dataset to `datasets/` as follows

```bash
# symlink the dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/openimages_v5c/
ln -s /vrd datasets/openimages_v5c/vrd
```

You can also prepare your own datasets.

Follow tsv dataset creation instructions [tools/mini_tsv/README.md](tools/mini_tsv/README.md)


### Single GPU training

```bash
python tools/train_sg_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the configuration files a global batch size that is divided over the number of GPUs. So if we only have a single GPU, this means that the batch size for that GPU will be 4x larger, which might lead to out-of-memory errors.


### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_sg_net.py --config-file "path/to/config/file.yaml" 
```


## Evaluation
You can test your model directly on single or multiple gpus. Here is an example on 4 GPUS:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vrd/R152FPN_vrd_reldn.yaml
```

## Abstractions
For more information on some of the main abstractions in our implementation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).

## Adding your own dataset

This implementation adds support for TSV style datasets.
But adding support for training on a new dataset can be done as follows:

```python
from maskrcnn_benchmark.data.datasets.relation_tsv import RelationTSVDataset

class MyDataset(RelationTSVDataset):
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
            is_load_label=True, **kwargs):

        super(MyDataset, self).__init__(yaml_file, extra_fields, transforms, is_load_label, **kwargs)
    
    def your_own_function(self, idx, call=False):
        # you can overwrite function or add your own functions this way
        pass
```
That's it. You can also add extra fields to the boxlist, such as segmentation masks
(using `structures.segmentation_mask.SegmentationMask`), or even your own instance type.

For a full example of how the `VGTSVDataset` is implemented, check [`maskrcnn_benchmark/data/datasets/vg_tsv.py`](maskrcnn_benchmark/data/datasets/vg_tsv.py).

Once you have created your dataset, it needs to be added in a couple of places:
- [`maskrcnn_benchmark/data/datasets/__init__.py`](maskrcnn_benchmark/data/datasets/__init__.py): add it to `__all__`
- [`maskrcnn_benchmark/data/datasets/utils/config_args.py`](maskrcnn_benchmark/data/datasets/utils/config_args.py): add it's name as an option to `tsv_dataset_name`


### Adding your own evaluation
To enable your dataset for testing, add a corresponding if statement in [`maskrcnn_benchmark/data/datasets/evaluation/__init__.py`](maskrcnn_benchmark/data/datasets/evaluation/__init__.py):
```python
if isinstance(dataset, datasets.MyDataset):
        return your_evaluation(**args)
```


## Feature extraction 
```bash
python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "../maskrcnn-benchmark-1/datasets1" TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True
```


## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```
@misc{TBD,
author = {TBD},
title = {{TBD}},
year = {2021},
howpublished = {\url{TBD}},
note = {Accessed: [Insert date here]}
}

```

  
## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement