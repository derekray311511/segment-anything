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
        self.P_POINT = 4
        self.N_POINT = 5
        self.BOXES = 6
        self.INFERENCE = 7
        self.UNDO = 8
        self.COLOR_MASKS = 9

MODE = Mode()

class SamAutoMaskGen:
    def __init__(self, model, args) -> None:
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
        self.amg_kwargs = self.get_amg_kwargs(args)
        self.generator = SamAutomaticMaskGenerator(model, output_mode=output_mode, **self.amg_kwargs)

    def get_amg_kwargs(self, args):
        amg_kwargs = {
            "points_per_side": args.points_per_side,
            "points_per_batch": args.points_per_batch,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "stability_score_offset": args.stability_score_offset,
            "box_nms_thresh": args.box_nms_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_nms_thresh": args.crop_nms_thresh,
            "crop_overlap_ratio": args.crop_overlap_ratio,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
        }
        amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
        return amg_kwargs

    def generate(self, image) -> np.ndarray:
        masks = self.generator.generate(image)
        np_masks = []
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            np_masks.append(mask)

        return np.array(np_masks, dtype=bool)

class SAM_Web_App:
    def __init__(self, args):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.args = args

        # load model
        print("Loading model...", end="")
        device = args.device
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.autoPredictor = SamAutoMaskGen(sam, args)
        print("Done")

        # Store the image globally on the server
        self.origin_image = None
        self.processed_img = None
        self.masked_img = None
        self.colorMasks = None
        self.imgSize = None
        self.imgIsSet = False           # To run self.predictor.set_image() or not

        self.mode = "p_point"           # p_point / n_point / box

        self.points = []
        self.points_label = []
        self.boxes = []
        self.masks = []

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
        self.colorMasks = np.zeros_like(image)
        self.imgSize = image.shape

        # Create image imbedding
        # self.predictor.set_image(image, image_format="RGB")   # Move to first inference

        # Reset inputs and masks and image ebedding
        self.imgIsSet = False
        self.reset_inputs()
        self.reset_masks()
        torch.cuda.empty_cache()

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
        self.points_label.append(1 if self.mode == 'p_point' else 0)

        # Process and return the image
        return f"Click at image pos {x}, {y}"
    
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
            elif (id == MODE.COLOR_MASKS):
                processed_image = self.colorMasks
            elif (id == MODE.CLEAR):
                processed_image = self.origin_image
                self.processed_img = self.origin_image
                self.reset_inputs()
                self.reset_masks()
            elif (id == MODE.P_POINT):
                self.mode = "p_point"
            elif (id == MODE.N_POINT):
                self.mode = "n_point"
            elif (id == MODE.BOXES):
                self.mode = "box"
            elif (id == MODE.INFERENCE):
                print("INFERENCE")
                self.reset_masks()
                points = np.array(self.points)
                labels = np.array(self.points_label)
                boxes = np.array(self.boxes)
                print(f"Points shape {points.shape}")
                print(f"Labels shape {labels.shape}")
                print(f"Boxes shape {boxes.shape}")
                processed_image = self.inference(self.origin_image, points, labels, boxes)
                self.get_colored_masks_image()
                self.processed_img = processed_image
            elif (id == MODE.UNDO):
                print("Undo")

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    def inference(self, image, points, labels, boxes) -> np.ndarray:

        points_len, lables_len, boxes_len = len(points), len(labels), len(boxes)
        if (len(points) == len(labels) == 0):
            points = labels = None
        if (len(boxes) == 0):
            boxes = None

        # Image is set ?
        if self.imgIsSet == False:
            self.predictor.set_image(image, image_format="RGB")
            self.imgIsSet = True
            print("Image set!")

        # Auto 
        if (points_len == boxes_len == 0):
            masks = self.autoPredictor.generate(image)
            for mask in masks:
                self.masks.append(mask)

        # One Object
        elif ((boxes_len == 1) or (points_len > 0 and boxes_len <= 1)):
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=boxes,
                multimask_output=True,
            )
            max_idx = np.argmax(scores)
            self.masks.append(masks[max_idx])

        # Multiple Object
        elif (boxes_len > 1):
            boxes = torch.tensor(boxes, device=self.predictor.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            max_idxs = np.argmax(scores, axis=1)
            print(f"output mask shape: {masks.shape}")  # (batch_size) x (num_predicted_masks_per_input) x H x W
            for i in range(masks.shape[0]):
                self.masks.append(masks[i][max_idxs[i]])

        # Update masks image to show
        masked_image = self.updateMaskImg(self.masks)

        return masked_image

    def updateMaskImg(self, masks):
        if (len(masks) == 0):
            return
        
        union_mask = np.zeros_like(masks[0])
        image = self.origin_image.copy()
        for mask in masks:
            image = self.overlay_mask(image, mask, 0.5, random_color=(len(masks) > 1))
            union_mask = np.bitwise_or(union_mask, mask)
        
        # Cut out objects using union mask
        masked_image = self.origin_image * union_mask[:, :, np.newaxis]
        self.masked_img = masked_image
        
        return image

    # Function to overlay a mask on an image
    def overlay_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float, 
        random_color: bool = False,
    ) -> np.ndarray:
        """ Draw mask on origin image

        parameters:
        image:  Origin image
        mask:   Mask that have same size as image
        color:  Mask's color in BGR
        alpha:  Transparent ratio from 0.0-1.0

        return:
        blended: masked image
        """
        # Blend the image and the mask using the alpha value
        if random_color:
            color = np.random.random(3)
        else:
            color = np.array([30/255, 144/255, 255/255])    # BGR
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask *= 255 * alpha
        mask = mask.astype(dtype=np.uint8)
        blended = cv2.add(image, mask)
        
        return blended
    
    def get_colored_masks_image(self):
        masks = self.masks
        image = self.colorMasks
        if (len(masks) == 0):
            return
        for mask in masks:
            image = self.overlay_mask(image, mask, 0.5, random_color=(len(masks) > 1))
        self.colorMasks = image
        return image
    
    def reset_inputs(self):
        self.points = []
        self.points_label = []
        self.boxes = []

    def reset_masks(self):
        self.masks = []
        self.masked_img = np.zeros_like(self.origin_image)
        self.colorMasks = np.zeros_like(self.origin_image)

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    args = parser().parse_args()
    app = SAM_Web_App(args)
    app.run(debug=True)
