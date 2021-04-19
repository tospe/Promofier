import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import argparse
import pytesseract

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./model/faster-rcnn-promos.pt",
				help="path to the model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.7, 
				help="confidence to keep predictions")
args = vars(ap.parse_args())

CLASS_NAMES = ["__background__", "product","name", "price", "promotion" ]
def get_prediction(img_path, confidence, device):
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
	img = Image.open(img_path).convert('RGB')
	transform = T.Compose([T.ToTensor()])
	img = transform(img).to(device)
	pred = model([img])
	pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
	pred_score = list(pred[0]['scores'].detach().cpu().numpy())

	pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]

	pred_boxes = pred_boxes[:pred_t+1]
	pred_class = pred_class[:pred_t+1]
	pred_score = pred_score[:pred_t+1]
	return pred_boxes, pred_class, pred_score
   
def detect_object(img_path, device, confidence=0.5, rect_th=2, text_size=1, text_th=4):
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
	boxes, pred_cls, pred_score = get_prediction(img_path, confidence, device)
	# boxes_sort = sorted(boxes, key = lambda x: x[0][0])
	names = [ i for i in range(len(boxes)) if pred_cls[i] == 'name' ]
	prices = [ i for i in range(len(boxes)) if pred_cls[i] == 'price' ]
	products = [ i for i in range(len(boxes)) if pred_cls[i] == 'product' ]

	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)
	text_color = (119, 76, 252)

	f = open("pred/products.txt", "w")
	for product in products:
		text = ""
		p_coord = boxes[product]
		for name in names:
			n_coord = boxes[name]
			if p_coord[0][0] <= n_coord[0][0] and p_coord[0][1] <= n_coord[0][1]:
				if p_coord[1][0] >= n_coord[1][0] and p_coord[1][1] >= n_coord[1][1]:
					img_crop = im_pil.crop( (n_coord[0][0], n_coord[0][1], n_coord[1][0], n_coord[1][1]) )
					p_name = str(pytesseract.image_to_string(img_crop, config='--psm 6'))
					p_name = p_name.replace("\n", " ")
					text = text + p_name
		for price in prices:
			price_coord = boxes[price]
			if p_coord[0][0] <= price_coord[0][0] and p_coord[0][1] <= price_coord[0][1]:
				if p_coord[1][0] >= price_coord[1][0] and p_coord[1][1] >= price_coord[1][1]:
					img_crop = im_pil.crop( (price_coord[0][0], price_coord[0][1], price_coord[1][0], price_coord[1][1]) )
					text = text + " " + str(pytesseract.image_to_string(img_crop, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789').strip() + "\n") 
		f.write(text)
	f.close()

	for i in range(len(boxes)):
		cv2.rectangle(img, boxes[i][0], boxes[i][1],color=text_color, thickness=rect_th)
		cv2.putText(img,pred_cls[i]+": "+ str(i)+ "  " +str(round(pred_score[i],3)), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color,thickness=text_th)
	fig = plt.figure()
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	fig.tight_layout()
	plt.savefig('pred/pred.png')
	plt.show()
	

if __name__ == "__main__":
	 
	if torch.cuda.is_available(): 
		device = torch.device('cuda')
		model = torch.load(args["model"])
	else: 
		device = torch.device('cpu')
		model = torch.load(args["model"], map_location=torch.device('cpu'))
		  
	
	img_path = args["image"]
	detect_object(img_path, device=device, confidence=args["confidence"])
	