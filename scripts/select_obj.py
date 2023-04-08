import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import cv2
import torch
import torchvision
import sys
import time

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

parser = argparse.ArgumentParser(
    description=("Select Object")
)
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="/home/ee904/DDFish/segment-anything/models/sam_vit_h_4b8939.pth")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model_type", type=str, default="default")
parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)
amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)
amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)
amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)
amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)
amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)
amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)
amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)
amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)
amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)
amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

class control:
    def __init__(self) -> None:
        # Mode
        self.mode = "point"             # use points or boxes
        self.prev_mode = None           # use to return previous mode when switching view mode
        self.view = "image"             # image that show on window
        # Point parameters
        self.prompt_type = "positive"   # prompt mode
        self.click_pos = []             # x, y
        self.click_label = []           # 0, 1
        # Box parameters
        self.boxes = []                 # Each box has [x1, y1, x2, y2]
        self.drawing = False            # drawing box or not
        self.start_point = None         # box start point

    # Define the callback function for handling mouse events
    def mouse_callback(self, event, x, y, flags, param):
        img = param         # Get the image from the 'param' argument
        H, W = img.shape[:2]
        circle_size = W // 300
        thickness = W // 500
        scale = W / 2000

        if self.mode == "point":

            if self.prompt_type == "positive":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_moving_pos(img, x, y, (W // 60), color, thickness, scale)

            elif event == cv2.EVENT_LBUTTONDOWN:
                # Perform the action when the left mouse button is clicked
                print(f"Mouse left clicked at ({x}, {y})")
                self.click_pos.append(np.array([x, y]))

                if self.prompt_type == "positive":
                    self.click_label.append(1)
                    cv2.circle(img, (x, y), circle_size, color, -1)  # Draw a red circle at the clicked position
                else:
                    self.click_label.append(0)
                    cv2.circle(img, (x, y), circle_size, color, -1)
                cv2.imshow('image', img)

        elif self.mode == "box":

            color = (224, 154, 22)

            if event == cv2.EVENT_MOUSEMOVE:
                if not self.drawing:
                    self.mouse_moving_pos(img, x, y, (W // 60), color, thickness, scale)
                else:
                    self.mouse_rect_moving(img, x, y, (W // 60), color, thickness, scale)

            elif event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing the bounding box
                self.drawing = True
                self.start_point = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Finish drawing the bounding box
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = (x, y)
                self.boxes.append([x1, y1, x2, y2])
                cv2.rectangle(img, self.start_point, (x, y), color, thickness)
                cv2.imshow('image', img)

        elif self.mode == "auto":

            color = (0, 255, 0)

            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_moving_pos(img, x, y, (W // 60), color, thickness, scale)

        elif self.mode == "view":

            color = (245, 183, 39)

            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_moving_pos(img, x, y, (W // 60), color, thickness, scale)
                    

    def mouse_moving_pos(self, img, x, y, shift, color, thickness=2, scale=0.5):
        # Display the mouse pointer coordinates on the image
        img_copy = img.copy()
        cv2.putText(img_copy, f"({self.mode.upper()})-({x}, {y})", (x+shift, y+shift), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        cv2.imshow('image', img_copy)

    def mouse_rect_moving(self, img, x, y, shift, color, thickness=2, scale=0.5):
        # Update the bounding box while holding the left mouse button
        img_copy = img.copy()
        cv2.putText(img_copy, f"({self.mode.upper()})-({x}, {y})", (x+shift, y+shift), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        cv2.rectangle(img_copy, self.start_point, (x, y), color, thickness)
        cv2.imshow('image', img_copy)

    def switch_view(self):
        if self.view == "masks":
            self.view = "image"
            self.mode = self.prev_mode

        elif self.view == "image":
            self.view = "masks"
            self.prev_mode = self.mode
            self.mode = "view"

    def reset(self):
        self.prompt_type = "positive"
        self.click_pos = []
        self.click_label = []
        self.boxes = []
        self.drawing = False
        self.start_point = None

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

# Function to overlay a mask on an image
def overlay_mask(
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

def main(args: argparse.Namespace):

    # load model
    print("Loading model...", end="")
    device = args.device
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    auto_predictor = SamAutoMaskGen(sam, args)
    print("Done")

    # load img
    BGR_origin_image = cv2.imread(args.img)
    image = cv2.cvtColor(BGR_origin_image, cv2.COLOR_BGR2RGB)

    # Process the image to produce an image embedding by calling SamPredictor.set_image. 
    # SamPredictor remembers this embedding and will use it for subsequent mask prediction.
    predictor.set_image(image)

    # Init the mouse event class
    mouse = control()
    init = True

    while (1):

        if init:
            # Create a window to display the image
            BGR_img = BGR_origin_image.copy()
            Object_img = np.zeros_like(BGR_origin_image)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # Set the initial window size (width, height)
            initial_size = (1440, 1080)
            cv2.resizeWindow('image', initial_size)
            # Init img and mouse callback
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            cv2.imshow('image', BGR_img)
            init = False

        # Wait for a key press
        key = cv2.waitKey(0)
        if key == 100 or key == 83: # d
            mouse.prompt_type = "negative"
            print("Click to add negative prompt")
            continue
        elif key == 97 or key == 81: # a
            mouse.prompt_type = "positive"
            print("Click to add positive prompt")
            continue
        elif key == 112: # p
            mouse.mode = "point"
            print("point mode")
            continue
        elif key == 98: # b
            mouse.mode = "box"
            print("box mode")
            continue
        elif key == 13: # enter
            mouse.mode = "auto"
            print("Auto segmentation")
            continue
        elif key == 118: # v
            mouse.switch_view()
            cv2.imshow('image', Object_img)
            cv2.setMouseCallback('image', mouse.mouse_callback, Object_img)
            print("Switch img/obj_img")
            while True:
                key = cv2.waitKey(0)
                if key == 118 or key == 27: 
                    break
            mouse.switch_view()
            cv2.imshow('image', BGR_img)
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            print("Switch img/obj_img")
            continue
        elif key == 32: # space
            print("SPACE: Inference")
        elif key == 27: # esc
            cv2.destroyAllWindows()
            break
        
        # Add prompt (points, boxes, )
        # To select the truck, choose a point on it. 
        # Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). 
        # Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.
        input_point = np.array(mouse.click_pos) if len(mouse.click_pos) > 0 else None
        input_label = np.array(mouse.click_label) if len(mouse.click_label) > 0 else None
        input_boxes = np.array(mouse.boxes) if len(mouse.boxes) > 0 else None
        if len(mouse.click_label) == 0 and len(mouse.boxes) == 0 and not (mouse.mode == "auto"):
            BGR_img = BGR_origin_image.copy()
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            cv2.imshow('image', BGR_img)
            print("Click on some objects first!")
            continue
        

        # With multimask_output=True (the default setting), SAM outputs 3 masks, where scores gives the model's own estimation of the quality of these masks. 
        # This setting is intended for ambiguous input prompts, and helps the model disambiguate different objects consistent with the prompt. 
        # When False, it will return a single mask. 
        # For ambiguous prompts such as a single point, it is recommended to use multimask_output=True even if only a single mask is desired; 
        # the best single mask can be chosen by picking the one with the highest score returned in scores. This will often result in a better mask.
        
        # Auto mode
        if (mouse.mode == "auto"):
            masks = auto_predictor.generate(image)
            BGR_img = BGR_origin_image.copy()
            merged_mask = np.zeros_like(masks[0])
            for i in range(masks.shape[0]):
                BGR_img = overlay_mask(BGR_img, masks[i], 0.5, random_color=True)
                merged_mask = np.bitwise_or(merged_mask, masks[i])
            Object_img = BGR_origin_image * merged_mask[:, :, np.newaxis]

        # One object prediction
        elif (len(mouse.boxes) <= 1):
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_boxes,
                multimask_output=True,
            )
            # print(masks.shape)  # (number_of_masks) x H x W

            max_idx = np.argmax(scores)
            max_score = scores[max_idx]

            # BGR image for cv2 to display
            BGR_img = overlay_mask(BGR_origin_image, masks[max_idx], 0.5, random_color=False)
            Object_img = BGR_origin_image * masks[max_idx][:, :, np.newaxis]

        # Multi-object prediction
        elif (len(mouse.boxes) > 1):
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            print(f"output mask shape: {masks.shape}")  # (batch_size) x (num_predicted_masks_per_input) x H x W

            max_idx = np.argmax(scores, axis=1)

            # BGR image for cv2 to display
            BGR_img = BGR_origin_image.copy()
            merged_mask = np.zeros_like(masks[0][0])
            for i in range(masks.shape[0]):
                BGR_img = overlay_mask(BGR_img, masks[i][max_idx[i]], 0.5, random_color=True)
                merged_mask = np.bitwise_or(merged_mask, masks[i][max_idx[i]])
            Object_img = BGR_origin_image * merged_mask[:, :, np.newaxis]
             
        # Set the mouse callback function for the window
        cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
        # Display the image
        cv2.imshow('image', BGR_img)
        # Reset input points and boxes
        mouse.reset()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
