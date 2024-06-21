from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO  # Assuming this is the library for YOLO
import cv2

app = Flask(__name__)




@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # Initialize YOLO model (adjust 'best_model.pt' as per your setup)
    model = YOLO('best_model.pt')

    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        annotated_image = results[0].plot()  # Generate annotated image
        annotated_image_path = os.path.join(UPLOAD_FOLDER, f'annotated_{filename}')
        cv2.imwrite(annotated_image_path, annotated_image)  # Save annotated image

        # Generate URL for annotated image
        annotated_image_url = url_for('static', filename=f'uploads/annotated_{filename}', _external=True)

        response_data = {
            'message': 'Object detection successful',
            'annotated_image': annotated_image_url,
            'object_counts': {}  # Provide object counts if available
        }

        return jsonify(response_data), 200
    else:
        return jsonify({'message': 'No objects detected or invalid results'}), 400

if _name_ == '__main__':
    app.run(debug=True)