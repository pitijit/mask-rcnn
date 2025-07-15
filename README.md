# mask-rcnn torchvision
Mask R-CNN is a deep learning model for instance segmentation — that means it can:\
-- Detect objects in an image (bounding box)\
-- Classify them (label: e.g., "cat", "car")\
-- Segment each object precisely at the pixel level (mask)\
It’s an extension of Faster R-CNN, with one extra branch for predicting segmentation masks.

Mask R-CNN has 3 main parts:\
-- Backbone: Feature extractor (e.g., ResNet)\
-- Region Proposal Network (RPN): Proposes object regions\
-- Heads:
   - Box head: classifies and refines bounding boxes
   - Mask head: predicts binary masks for each object

### dataset structure
```
├── pipe_4c/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── train_coco.json
│       └── val_coco.json
```
