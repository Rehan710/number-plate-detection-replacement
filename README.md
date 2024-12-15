**Number Plate Detection, Replacement, and Text Extraction**

This project allows users to upload images of vehicles and automatically detect, replace, and extract text from number plates. It uses machine learning models for object detection and OpenCV with Tesseract for text extraction. The detected number plates are replaced with a logo, and the extracted text is saved for further analysis.

**Features**
•	Number Plate Detection: Detects vehicle number plates in uploaded images.

•	Text Extraction: Extracts the text from detected number plates using OCR.
•	Logo Replacement: Replaces detected number plates with a custom logo.
•	Manual Annotation: If no number plates are detected, users can manually annotate the bounding boxes.
•	Data Export: Saves extracted data (text and bounding boxes) to a CSV file.


**Technologies Used**
•	Streamlit: Web app framework for building the user interface.
•	YOLOv5: Pre-trained model for number plate detection.
•	OpenCV: Image processing and contour extraction.
•	Tesseract OCR: Optical character recognition for text extraction.
•	PIL: Image handling and manipulation.
•	Pandas: Data storage and export to CSV.


**Installation**
1.	Clone this repository:
git clone https://github.com/your-username/number-plate-detection.git
2.	Navigate to the project directory:

cd number-plate-detection
3.	Install the required dependencies:
pip install -r requirements.txt



**Usage**
1. Run the Streamlit Application
Launch the app by running the following command:
streamlit run app.py

2. **Upload Image**
Once the application is running, you will be prompted to upload an image file (PNG, JPG, JPEG). The image will be processed to detect number plates.

**3. Automatic Number Plate Detection and Replacement**
If the model detects a number plate:
•	It will show the bounding boxes around the detected number plates.
•	The detected number plates will be replaced with a logo.
•	The text from the number plates will be extracted and displayed in a table.


**4. Manual Annotation**
If no number plates are detected:
•	You will be prompted to annotate the bounding boxes manually using a drawing canvas.
•	After annotating, you can extract the text from the number plates and replace them with a logo.


**Files and Directories**
•	app.py: Main Streamlit app to run the interface.
•	model.py: Model code for object detection and text extraction.
•	logo_path: Path to the logo image that replaces detected number plates.
•	output_dir: Directory to save the modified images.
•	annotation_dir: Directory to save manual annotations.
•	text_output_file: Path to save the extracted text data in CSV format.


**Requirements**
•	Python 3.x
•	Streamlit
•	OpenCV
•	YOLOv5 pre-trained model (or another model for object detection)
•	Tesseract OCR
•	Pandas
•	Pillow
Install dependencies with:
pip install -r requirements.txt







