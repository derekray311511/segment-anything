# SAM - Local APP

## DEMO video
[![cv2 DEMO](https://img.youtube.com/vi/rCMvSxbb5Lc/0.jpg)](https://www.youtube.com/watch?v=rCMvSxbb5Lc)

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

We have tested:
`Python 3.8`
`pytorch 2.0.0 (py3.8_cuda11.7_cudnn8.5.0_0)`
`torchvision 0.15.0`

<!--PyQt5 version history Successfully installed PyQt5-Qt5-5.15.2 PyQt5-sip-12.11.1 pyqt5-5.15.9 -->

Install Segment Anything:
```bash
git@github.com:derekray311511/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. jupyter is also required to run the example notebooks.
```bash!
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Model Checkpoints
You can download the model checkpoints [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Functions

#### Mode
- `Auto`: Segment all objects in the image
- `Custom`: Select object(s) with points or boxes using mouse clicks
- `View`: View the masks you just created and disable manipulation

#### Auto mode
- Press `SPACE` to inference all objects in the image

#### Custom mode
- `Point select`: Press `p` to switch to `point select` function
    - `a`: Positive prompt
    - `d`: Negative prompt
- `Box select`: Press `b` to switch to `box select` function

#### View mode
- Press `v` to switch between view / previous mode

#### Shortcut Table

|      Function       |     Key    |
| ----------          | ---------- |
|Switch to auto mode  |    enter   |
|Switch to view mode  |      v     |
|Point select mode    |      p     |
|Box select mode      |      b     |
|Positive prompt      |      a     |
|Negative prompt      |      d     |
|Save image           |      s     |
|Inference            |    SPACE   |

