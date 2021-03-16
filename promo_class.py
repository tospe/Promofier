import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import xml.etree.ElementTree as ET 

CLASS_NAMES = ["__background__", "product","name", "price", "promotion" ]
class PromotionDataset(torch.utils.data.Dataset):
	def __init__(self,data_dir,transforms=None):
		self.data_dir = data_dir
		self.transforms = transforms
		self.imgs = list(sorted(os.listdir( data_dir )))
		self.annotations = list(sorted(os.listdir( data_dir )))

	def __getitem__(self,idx):
		#load images and annotations
		img_path = os.path.join(self.data_dir, "imgs", self.imgs[idx])
		img = Image.open(img_path).convert("RGB")

		#annots
		annot_path = os.path.join(self.data_dir, "annotations", self.annotations[idx])
		annots = ET.parse(annot_path)
		objects = annots.getroot().findall('object')
		num_objs = len(objects)
		boxes = []
		labels = []

		#get all boxes
		for o in objects:
			xmin = int(o.find('bndbox').find('xmin').text)
			xmax = int(o.find('bndbox').find('xmax').text)
			ymin = int(o.find('bndbox').find('ymin').text)
			ymax = int(o.find('bndbox').find('ymax').text)
			boxes.append([xmin, ymin, xmax, ymax])

		boxes = torch.as_tensor(boxes, dtype=torch.float32) 	
		labels = torch.ones((num_objs,), dtype=torch.int64)
		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

		# suppose all instances are not crowd
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target  

	def __len__(self):
		return len(self.imgs)        