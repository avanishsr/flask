from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('besttrainptn.pt')

# Define the upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process and display results for each image
def process_image(image_path):
    img = cv2.imread(image_path)

    # Get predictions with the specified confidence threshold
    results = model.predict(source=image_path)
    damaged_parts = []
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()
        masks = result.masks  # Do not call cpu() or numpy() yet

        for box, score, cls in zip(boxes, scores, classes):
            # Draw bounding box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, f'{model.names[int(cls)]} {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if masks is not None
            if masks is not None:
                mask = masks.cpu().numpy()
                # Draw segmentation mask
                mask = np.array(mask).astype(np.uint8)
                color_mask = np.zeros_like(img, dtype=np.uint8)
                color_mask[mask == 1] = (0, 255, 0)  # Set mask color to green
                img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)

            damaged_parts.append({
                'class': model.names[int(cls)],
                'score': float(score),  # Convert score to Python float type
                'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })

    # Convert the processed image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return damaged_parts, img_base64

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    damaged_parts, img_base64 = process_image(image_path)

    return jsonify({
        'damaged_parts': damaged_parts,
        'processed_image_base64': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
