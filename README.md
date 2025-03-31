
# Invoice OCR using Object Detection

A python project to extract textual information from invoices using PaddleOCR for OCR & Yolov11s for the extraction of region of interest (bounding boxes of fields).  

Yolov11s was trained on dataset of invoices labelled on 18 classes— Discount_Percentage, Due_Date, Email_Client, Name_Client, Products, Remise, Subtotal, Tax, Tax_Precentage, Tel_Client, billing address, invoice date, invoice number, shipping address, supplier-address, supplier-name, supplier-phone, total.





## Features

- Performs Object Detection followed by Text Extraction.
- Outputs the extracted text in a structual format (JSON).
- The extracted text is mapped to its corresponding class defined in the dataset.
- This project was run on Test files (dataset/test/images), and the output is stored in JSON_output folder.



## Usage

Clone the project

```bash
  git clone https://github.com/komalhari7/invoice_OCR
```

Go to the project directory

```bash
  cd invoice_OCR
```

Install required packages

```bash
  pip install ultralytics
  pip install paddlepaddle
  pip install paddleOCR
```

Change the path to required directory inside detect.py

```bash
  folder_path = "dataset//test//images"
```
Run the script

```bash
  python detect.py
```
