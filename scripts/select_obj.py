import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import cv2
import torch
import torchvision
import sys

from segment_anything import sam_model_registry, SamPredictor

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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

class control:
    def __init__(self) -> None:
        # Mode
        self.mode = "point"             # use points or boxes
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

        if self.mode == "point":

            if self.prompt_type == "positive":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_moving_pos(img, x, y, (W // 50), color, 2)

            elif event == cv2.EVENT_LBUTTONDOWN:
                # Perform the action when the left mouse button is clicked
                print(f"Mouse left clicked at ({x}, {y})")
                circle_size = W // 300
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
                    self.mouse_moving_pos(img, x, y, (W // 50), color, 2)
                else:
                    self.mouse_rect_moving(img, x, y, (W // 50), color, 2)

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
                cv2.rectangle(img, self.start_point, (x, y), color, 2)
                cv2.imshow('image', img)
                    

    def mouse_moving_pos(self, img, x, y, shift, color, thickness=2):
        # Display the mouse pointer coordinates on the image
        img_copy = img.copy()
        cv2.putText(img_copy, f"({x}, {y})", (x+shift, y+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        cv2.imshow('image', img_copy)

    def mouse_rect_moving(self, img, x, y, shift, color, thickness=2):
        # Update the bounding box while holding the left mouse button
        img_copy = img.copy()
        cv2.putText(img_copy, f"({x}, {y})", (x+shift, y+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        cv2.rectangle(img_copy, self.start_point, (x, y), color, 2)
        cv2.imshow('image', img_copy)

    def reset(self):
        self.prompt_type = "positive"
        self.click_pos = []
        self.click_label = []
        self.boxes = []
        self.drawing = False
        self.start_point = None

# Function to overlay a mask on an image
def overlay_mask(image, mask, alpha):
    # Blend the image and the mask using the alpha value
    color = np.array([30/255, 144/255, 255/255])    # BGR
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask *= 255 * alpha
    mask = mask.astype(dtype=np.uint8)
    blended = cv2.add(image, mask)
    return blended


def main(args: argparse.Namespace):
    # load img
    BGR_origin_image = cv2.imread(args.img)
    image = cv2.cvtColor(BGR_origin_image, cv2.COLOR_BGR2RGB)

    # load model
    device = args.device
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

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
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            print("D: Click to add negative prompt")
            continue
        elif key == 97 or key == 81: # a
            mouse.prompt_type = "positive"
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            print("A: Click to add positive prompt")
            continue
        elif key == 112: # p
            mouse.mode = "point"
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            print("point mode")
            continue
        elif key == 98: # b
            mouse.mode = "box"
            cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)
            print("box mode")
            continue
        elif key == 13: # enter
            print("Enter")
        elif key == 32: # space
            print("SPACE: Inference")
        elif key == 27: # esc
            cv2.destroyAllWindows()
            break
        
        # Add prompt (points, boxes, )
        # To select the truck, choose a point on it. 
        # Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). 
        # Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.
        input_point = None
        input_label = None
        input_boxes = None
        if len(mouse.click_label) != 0:
            input_point = np.array(mouse.click_pos)
            input_label = np.array(mouse.click_label)
        if len(mouse.boxes) != 0:
            input_boxes = np.array(mouse.boxes)
        if len(mouse.click_label) == 0 and len(mouse.boxes) == 0:
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
        
        # One object prediction
        if (len(mouse.boxes) <= 1):
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
            BGR_img = overlay_mask(BGR_origin_image, masks[max_idx], 0.5)
        
        # Multi-object prediction
        else:
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
            print(masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W

            max_idx = np.argmax(scores, axis=1)

            # BGR image for cv2 to display
            BGR_img = BGR_origin_image.copy()
            for i in range(masks.shape[0]):
                BGR_img = overlay_mask(BGR_img, masks[i][max_idx[i]], 0.5)

        # Set the mouse callback function for the window
        cv2.setMouseCallback('image', mouse.mouse_callback, BGR_img)

        # Display the image
        cv2.imshow('image', BGR_img)

        # Reset input points
        mouse.reset()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)