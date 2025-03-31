# pip install ultralytics
# pip install paddlepaddle
# pip install paddleOCR

from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw
import numpy as np
import json
import os

ocr = PaddleOCR(lang="en")  
# Load a model
model = YOLO("yolo11s.pt")  # load an official model
model = YOLO("best.pt")  # load a custom model

#Define folder path & targeted files
folder_path = "dataset//test//images"
files = os.listdir(folder_path) 


#Loop through each invoies
for index1, value1 in enumerate(files):

    #path to the invoice
    path = os.path.join(folder_path, value1)
     

    # perform field detection using Yolov11s 
    results = model(path)  

    #Grab the result
    result = results[0]
   
    detection = []
    for index2, value2 in enumerate(result.boxes.cls.int().tolist()):
        detection.append({"class" : result.names[value2],
                        "conf_score" : result.boxes.conf.tolist()[index2],
                        "bbox" : result.boxes.xyxy.tolist()[index2]
                        })


    # Draw OCR results
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)

    final_output = []
    for index3, value3 in enumerate(detection):
        # draw.rectangle(value["bbox"], outline="red", width=3) #for bbox
        cropped_image = image.crop(value3["bbox"])
        cropped_image_np = np.array(cropped_image)
        ocr_output = ocr.ocr(cropped_image_np)
        if(ocr_output[0] != None):
            text=[line[1][0] for line in ocr_output[0]]
            final_output.append({
            value3["class"] : text
        })   
        

    # Define output JSON filename (same as image but with .json)
    json_filename = f"{value1}.json"
    os.makedirs("JSON_output", exist_ok=True)
    json_path = os.path.join("JSON_output", json_filename)

    # Write to JSON file
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_output, json_file, indent=4)



