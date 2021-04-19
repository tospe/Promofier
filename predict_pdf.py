import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import os

import cv2
import random
import argparse
import pytesseract
from wand.image import Image as wImage
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./model/faster-rcnn-promos.pt",
				help="path to the model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.7, 
				help="confidence to keep predictions")
args = vars(ap.parse_args())

CLASS_NAMES = ["__background__", "product","name", "price", "promotion" ]
def get_pdf_prediction(confidence, device):
	"""
	get_prediction
	  parameters:
		- img_path - path of the input image
		- confidence - threshold value for prediction score
	  method:
		- Image is obtained from the image path
		- the image is converted to image tensor using PyTorch's Transforms
		- image is passed through the model to get the predictions
		- class, box coordinates are obtained, but only prediction score > threshold
		  are chosen.
	
	"""
	predictions_file = "pred/all.txt"
	total_products = 0
	files = [ file for file in os.listdir("./test-pdfs") if file.endswith(".jpg") ]

	if os.path.exists(predictions_file):
		os.remove(predictions_file)
	for file in files:
		print(file)
		#prepare prediction
		im_pil = Image.open(os.path.join("test-pdfs",file)).convert('RGB')
		transform = T.Compose([T.ToTensor()])
		img = transform(im_pil).to(device)

		#prediction
		pred = model([img])
		pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
		pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
		pred_score = list(pred[0]['scores'].detach().cpu().numpy())
		pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
		if len(pred_t) < 1:
			os.remove(os.path.join("test-pdfs",file))
			continue
		pred_t = pred_t[-1]
		pred_boxes = pred_boxes[:pred_t+1]
		pred_class = pred_class[:pred_t+1]
		pred_score = pred_score[:pred_t+1]
	
		names = [ i for i in range(len(pred_boxes)) if pred_class[i] == 'name' ]
		prices = [ i for i in range(len(pred_boxes)) if pred_class[i] == 'price' ]
		products = [ i for i in range(len(pred_boxes)) if pred_class[i] == 'product' ]		

		with open(predictions_file, "a+") as f:
			for product in products:
				total_products += 1
				text = ""
				p_coord = pred_boxes[product]
				for name in names:
					n_coord = pred_boxes[name]
					if p_coord[0][0] <= n_coord[0][0] and p_coord[0][1] <= n_coord[0][1]:
						if p_coord[1][0] >= n_coord[1][0] and p_coord[1][1] >= n_coord[1][1]:
							img_crop = im_pil.crop( (n_coord[0][0], n_coord[0][1], n_coord[1][0], n_coord[1][1]) )
							p_name = str(pytesseract.image_to_string(img_crop, config='--psm 6'))
							p_name = p_name.replace("\n", " ")
							text = text + p_name
				for price in prices:
					price_coord = pred_boxes[price]
					if p_coord[0][0] <= price_coord[0][0] and p_coord[0][1] <= price_coord[0][1]:
						if p_coord[1][0] >= price_coord[1][0] and p_coord[1][1] >= price_coord[1][1]:
							img_crop = im_pil.crop( (price_coord[0][0], price_coord[0][1], price_coord[1][0], price_coord[1][1]) )
							text = text + " " + str(pytesseract.image_to_string(img_crop, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789').strip() + "\n") 
				f.write(text)
			os.remove(os.path.join("test-pdfs",file))
	return pred_boxes, pred_class, pred_score	


def prepare_pdf(pdf_path):
	with(wImage(filename=pdf_path, resolution=120)) as source: 
		for i, image in enumerate(source.sequence):
			print("image", str(i))
			newfilename = pdf_path[:-4] + str(i) +'.jpg'
			wImage(image).save(filename=newfilename)	

			
   
def detect_object(pdf_path, device, confidence=0.5, rect_th=2, text_size=1, text_th=4):
	"""
	object_detection_api
	  parameters:
		- img_path - path of the input image
		- confidence - threshold value for prediction score
		- rect_th - thickness of bounding box
		- text_size - size of the class label text
		- text_th - thichness of the text
	  method:
		- prediction is obtained from get_prediction method
		- for each prediction, bounding box is drawn and text is written 
		  with opencv
		- the final image is displayed
	"""
	print("preparing")
	# prepare_pdf(pdf_path)
	print("predicting")
	boxes, pred_cls, pred_score = get_pdf_prediction(confidence, device)
	
	

if __name__ == "__main__":
	 
	if torch.cuda.is_available(): 
		device = torch.device('cuda')
		model = torch.load(args["model"])
	else: 
		device = torch.device('cpu')
		model = torch.load(args["model"], map_location=torch.device('cpu'))
		  
	
	img_path = args["image"]
	detect_object(img_path, device=device, confidence=args["confidence"])
	