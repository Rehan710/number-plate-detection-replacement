**Number Plate Detection, Replacement, and Text Extraction**

This project allows users to upload images of vehicles and automatically detect, replace, and extract text from number plates. It uses machine learning models for object detection and OpenCV with Tesseract for text extraction. The detected number plates are replaced with a logo, and the extracted text is saved for further analysis.

**Features**<br/>
•	Number Plate Detection: Detects vehicle number plates in uploaded images.<br/>
•	Text Extraction: Extracts the text from detected number plates using OCR.<br/>
•	Logo Replacement: Replaces detected number plates with a custom logo.<br/>
•	Manual Annotation: If no number plates are detected, users can manually annotate the bounding boxes.<br/>
•	Data Export: Saves extracted data (text and bounding boxes) to a CSV file.<br/>


**Technologies Used**<br/>
•	Streamlit: Web app framework for building the user interface.<br/>
•	YOLOv5: Pre-trained model for number plate detection.<br/>
•	OpenCV: Image processing and contour extraction.<br/>
•	Tesseract OCR: Optical character recognition for text extraction.<br/>
•	PIL: Image handling and manipulation.<br/>
•	Pandas: Data storage and export to CSV.<br/>


**Installation**<br/>
1.	Clone this repository:<br/>
git clone https://github.com/your-username/number-plate-detection.git
2.	Navigate to the project directory:<br/>
cd number-plate-detection<br/>
3.	Install the required dependencies:<br/>
```cmd
pip install -r requirements.txt
```



**Usage**<br/>
1. Run the Streamlit Application<br/>
Launch the app by running the following command:<br/>
```cmd
streamlit run app.py
```

2. **Upload Image**<br/>
Once the application is running, you will be prompted to upload an image file (PNG, JPG, JPEG). The image will be processed to detect number plates.<br/>

**3. Automatic Number Plate Detection and Replacement**<br/>
If the model detects a number plate:<br/>
•	It will show the bounding boxes around the detected number plates.<br/>
•	The detected number plates will be replaced with a logo.<br/>
•	The text from the number plates will be extracted and displayed in a table.<br/>


**4. Manual Annotation**<br/>
If no number plates are detected:<br/>
•	You will be prompted to annotate the bounding boxes manually using a drawing canvas.<br/>
•	After annotating, you can extract the text from the number plates and replace them with a logo.<br/>


**Files and Directories**<br/>
•	app.py: Main Streamlit app to run the interface.<br/>
•	model.py: Model code for object detection and text extraction.<br/>
•	logo_path: Path to the logo image that replaces detected number plates.<br/>
•	output_dir: Directory to save the modified images.<br/>
•	annotation_dir: Directory to save manual annotations.<br/>
•	text_output_file: Path to save the extracted text data in CSV format.<br/>


**Requirements**<br/>
•	Python 3.x<br/>
•	Streamlit<br/>
•	OpenCV<br/>
•	YOLOv5 pre-trained model (or another model for object detection)<br/>
•	Tesseract OCR<br/>
•	Pandas<br/>
•	Pillow<br/>







