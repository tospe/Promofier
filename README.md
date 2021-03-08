# Fine-Tune Faster-RCNN on a custom promotions flyer dataset using pytorch

Goal: Identify products in a promotions flyer.


## Usage

__Train the model__
```shell
python3 train.py
```

The model will be3 `./model/faster-rcnn-promos.pt`

__Model Inference__

```shell
python3 predict.py --image path/to/test/image

#for example
python3 predict.py --image beagle/val/00000184.jpg
```
__Note__: `utils.py`, `transforms.py`, `coco_eval.py`, `coco_utils.py`, `engine.py` contains helper functions used during training process, and they are adopted from PyTorch Repo.