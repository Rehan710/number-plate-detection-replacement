import streamlit as st
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
from PIL import Image, ImageDraw
import easyocr
import pytesseract
import os
import cv2
import numpy as np
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directories
output_dir = r"D:\Jupyter notebook\assesment\Dataset\annotated_images"
annotation_dir = r"D:\Jupyter notebook\assesment\Dataset\annotations"
text_output_file = r"D:\Jupyter notebook\assesment\Dataset\extracted_text.csv"
logo_path = r"D:\Jupyter notebook\assesment\Dataset\New folder\logo.png"

metrics_dir = []

# Create directories if not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)

# Load YOLO model
model = YOLO(r"D:\Jupyter notebook\assesment\10_epochs\best.pt")


# File paths for metrics images
METRICS_IMAGES = {
    "Confusion Matrix": r"D:\Jupyter notebook\assesment\run\confusion_matrix.png",
    "Normalized Confusion Matrix": r"D:\Jupyter notebook\assesment\run\confusion_matrix_normalized.png",
    "F1 Curve": r"D:\Jupyter notebook\assesment\run\F1_curve.png",
    "Labels Distribution": r"D:\Jupyter notebook\assesment\run\labels.jpg",
    "P Curve": r"D:\Jupyter notebook\assesment\run\P_curve.png",
    "PR Curve": r"D:\Jupyter notebook\assesment\run\PR_curve.png",
    "R Curve": r"D:\Jupyter notebook\assesment\run\R_curve.png",
    "Results Summary": r"D:\Jupyter notebook\assesment\run\results.png",
}


def detect_and_extract_text(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # Focus on top contours
    extracted_texts = []

    for c in cnts:
        area = cv2.contourArea(c)
        if 800 > area > 200:  # Adjust area range for text regions
            x, y, w, h = cv2.boundingRect(c)

            # Crop the region of interest (ROI)
            ROI = gray[y:y+h, x:x+w]
            ROI_resized = cv2.resize(ROI, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ROI_inverted = 255 - ROI_resized

            # Use pytesseract to extract text from the cropped region
            text = pytesseract.image_to_string(ROI_inverted, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if text.strip():
                extracted_texts.append({"Bounding Box": (x, y, w, h), "Extracted Text": text.strip()})

    return extracted_texts, mask, thresh


def extract_text_with_contours(image, bbox):
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    cropped_image = image.crop((x1, y1, x2, y2))  # Crop the number plate region

    # Convert to grayscale
    gray_image = cropped_image.convert("L")
    gray_image = np.array(gray_image)

    # Apply thresholding
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    extracted_text = ""
    for c in contours:
        # Extract the bounding box for each contour
        area = cv2.contourArea(c)
        if area > 200:  # Filter out smaller contours
            x, y, w, h = cv2.boundingRect(c)
            ROI = thresholded_image[y:y + h, x:x + w]

            # Use Tesseract to extract text from the region
            text = pytesseract.image_to_string(ROI,
                                               config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if text.strip():
                extracted_text = text.strip()
                break  # Return the first non-empty text found

    return extracted_text


# Streamlit App
st.title("Number Plate Detection, Replacement, and Text Extraction")
tab1, tab2 = st.tabs(["Detection", "Metrics"])

# Detection Tab
with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform object detection
        st.write("Detecting number plate...")
        results = model(img)
        detected = False
        extracted_data = []

        for result in results:
            boxes = result.boxes  # Detected bounding boxes
            if len(boxes) > 0:
                detected = True
                st.write(f"Detected {len(boxes)} number plate(s)")
                draw = ImageDraw.Draw(img)

                # Replace each detected number plate with the logo and extract text
                logo = Image.open(logo_path)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    st.write(f"Bounding Box: {x1}, {y1}, {x2}, {y2}")

                    # Extract text from the detected number plate using contours and pytesseract
                    text = extract_text_with_contours(img, (x1, y1, x2, y2))
                    extracted_data.append(
                        {"File Name": uploaded_file.name, "Bounding Box": (x1, y1, x2, y2), "Extracted Text": text})

                    # Replace number plate with logo
                    logo_resized = logo.resize((x2 - x1, y2 - y1))
                    img.paste(logo_resized, (x1, y1))

                # Save modified image
                output_image_path = os.path.join(output_dir, f"output_{uploaded_file.name}")
                img.save(output_image_path)
                st.image(img, caption="Modified Image with Logo", use_column_width=True)
                st.success(f"Modified image saved at: {output_image_path}")

        # If no detections, allow manual annotation
        if not detected:
            st.warning("No detections found. Annotate the number plate manually.")

            # Canvas for manual annotation
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red
                stroke_width=2,
                background_image=img,
                height=img.height,
                width=img.width,
                drawing_mode="rect",  # Rectangle for bounding box
                key="canvas",
            )

            # Save annotation
            if st.button("Save Annotation"):
                if canvas_result.json_data is not None:
                    for obj in canvas_result.json_data["objects"]:
                        left = int(obj["left"])
                        top = int(obj["top"])
                        width = int(obj["width"])
                        height = int(obj["height"])
                        x1, y1, x2, y2 = left, top, left + width, top + height
                        st.write(f"Bounding Box: {x1}, {y1}, {x2}, {y2}")

                        # Save annotation to .txt file
                        annotation_file = os.path.join(annotation_dir, uploaded_file.name.replace('.png', '.txt'))
                        with open(annotation_file, "w") as f:
                            f.write(f"{x1},{y1},{x2},{y2}")

                        # Extract text from manually annotated bounding box using contours
                        text = extract_text_with_contours(img, (x1, y1, x2, y2))
                        extracted_data.append(
                            {"File Name": uploaded_file.name, "Bounding Box": (x1, y1, x2, y2), "Extracted Text": text})

                        # Replace number plate with logo
                        logo = Image.open(logo_path)
                        logo_resized = logo.resize((width, height))
                        img.paste(logo_resized, (x1, y1))

                    # Save modified image
                    output_image_path = os.path.join(output_dir, f"annotated_{uploaded_file.name}")
                    img.save(output_image_path)
                    st.image(img, caption="Modified Image with Logo (Annotated)", use_column_width=True)
                    st.success(f"Annotated image saved at: {output_image_path}")
                else:
                    st.warning("No annotation found. Please draw a bounding box.")

        # Save extracted data to a CSV file
        if extracted_data:
            df = pd.DataFrame(extracted_data)
            st.table(df)
            df.to_csv(text_output_file, index=False)
            st.success(f"Extracted text saved at: {text_output_file}")


# Tab 2: Model Metrics
with tab2:
    st.header("Model Metrics and Visualizations")
    st.write("Below are the key performance metrics and visualizations for the model.")

    # Display each metric image with description
    for metric_name, image_path in METRICS_IMAGES.items():
        st.subheader(metric_name)
        try:
            img = Image.open(image_path)
            st.image(img, caption=metric_name, use_column_width=True)
        except FileNotFoundError:
            st.warning(f"File not found: {image_path}")

# Footer
st.write("Ensure the image files and model weights are stored at the specified paths.")
