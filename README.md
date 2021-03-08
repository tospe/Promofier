# Fine-Tune Faste-RCNN on a custom promotions flyer dataset using pytorch

Goal: Identify products in a promotions flyer.


## Usage

__Train the model__
```shell
python3 train.py
```
The trained model will be saved in the `output/` with name `faster-rcnn-beagle.pt`

__Model Inference__

```shell
python3 predict.py --image path/to/test/image

#for example
python3 predict.py --image beagle/val/00000184.jpg
```
__Note__: `utils.py`, `transforms.py`, `coco_eval.py`, `coco_utils.py`, `engine.py` contains helper functions used during training process, and they are adopted from PyTorch Repo.