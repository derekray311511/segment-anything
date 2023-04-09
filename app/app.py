from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from typing import Any, Dict, List
from arg_parse import parser
import cv2
import numpy as np
import io, os
import time
import base64
import argparse
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

class Mode:
    def __init__(self) -> None:
        self.IAMGE = 1
        self.MASKS = 2
        self.CLEAR = 3
        self.POINT = 4
        self.BOXES = 5
        self.INFERENCE = 6

MODE = Mode()

class ImageProcessingApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        # Store the image globally on the server
        self.origin_image = None
        self.processed_img = None
        self.masked_img = None
        self.imgSize = None
        self.mode = "point"

        self.points = []
        self.boxes = []

        self.app.route('/', methods=['GET'])(self.home)
        self.app.route('/upload_image', methods=['POST'])(self.upload_image)
        self.app.route('/button_click', methods=['POST'])(self.button_click)
        self.app.route('/point_click', methods=['POST'])(self.handle_mouse_click)
        self.app.route('/box_receive', methods=['POST'])(self.box_receive)

    def home(self):
        return render_template('index.html')

    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request'}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Store the image globally
        self.origin_image = image
        self.processed_img = image
        self.masked_img = np.zeros_like(image)
        self.imgSize = image.shape

        return "Uploaded image, successfully initialized"

    def button_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        button_id = data['button_id']
        print(f"Button {button_id} clicked")

        # Info
        info = {
            'event': 'button_click',
            'data': button_id
        }

        # Process and return the image
        return self.process_image(self.processed_img, info)

    def handle_mouse_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        x = data['x']
        y = data['y']
        print(f'Point clicked at: {x}, {y}')
        self.points.append(np.array([x, y], dtype=np.float32))

        # Info
        info = {
            'event': 'point_click',
            'data': np.array([x, y])
        }

        # Process and return the image
        return self.process_image(self.processed_img, info)
    
    def box_receive(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        self.boxes.append(np.array([
            data['x1'], data['y1'],
            data['x2'], data['y2']
        ], dtype=np.float32))

        return "server received boxes"

    def process_image(self, image, info):
        processed_image = image

        if info['event'] == 'button_click':
            id = info['data']
            if (id == MODE.IAMGE):
                processed_image = self.processed_img
            elif (id == MODE.MASKS):
                processed_image = self.masked_img
            elif (id == MODE.CLEAR):
                processed_image = self.origin_image
                self.reset_inputs()
            elif (id == MODE.POINT):
                self.mode = "point"
            elif (id == MODE.BOXES):
                self.mode = "box"
            elif (id == MODE.INFERENCE):
                print("INFERENCE")
                points = np.array(self.points)
                boxes = np.array(self.boxes)
                print(f"Points shape {points.shape}")
                print(f"Boxes shape {boxes.shape}")

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    def reset_inputs(self):
        self.points = []
        self.boxes = []

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    args = parser().parse_args()
    app = ImageProcessingApp()
    app.run(debug=True)
