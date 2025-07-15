# ==== Import Libraries ====
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F

from engine import train_one_epoch, evaluate
import utils
from config import TrainConfig
from transforms import get_transform
from pycocotools import mask as coco_mask


# ==== Custom COCO Dataset ====
class CocoInstanceDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image and annotations
        img, annotations = super().__getitem__(idx)
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        image_id = img_info["id"]

        # Get original image size
        if hasattr(img, 'size') and isinstance(img.size, tuple):
            width, height = img.size
        elif isinstance(img, torch.Tensor):
            _, height, width = img.shape
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Extract bounding boxes, labels, masks
        boxes, labels, masks = [], [], []
        for obj in annotations:
            if 'bbox' not in obj:
                continue
            xmin, ymin, w, h = obj['bbox']
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

            if 'segmentation' in obj:
                rle = coco_mask.frPyObjects(obj['segmentation'], height, width)
                m = coco_mask.decode(rle)
                if len(m.shape) == 3:
                    m = m.any(axis=2)  # merge multiple segments
                masks.append(m)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(image_id)}

        # Convert masks if available
        if masks:
            masks = np.array(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            target["masks"] = masks

        # Print shape info (debugging)
        #print(f"boxes shape: {target['boxes'].shape}, labels shape: {target['labels'].shape}")
        #if "masks" in target:
            #print(f"masks shape: {target['masks'].shape}")

        # Apply transforms
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


# ==== Model Loader ====
def get_instance_segmentation_model(num_classes):
    # Load pretrained weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# ==== Visualization Function ====
def show_prediction(model, dataset, device, save_dir, index=None, score_threshold=0.5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Select a random image if not specified
    index = index if index is not None else random.randint(0, len(dataset) - 1)
    img, _ = dataset[index]
    img = img.to(device)

    with torch.no_grad():
        prediction = model([img])[0]

    # Convert tensors to numpy
    img = img.cpu()
    masks = prediction['masks'].cpu().squeeze(1).detach().numpy()
    boxes = prediction['boxes'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()

    # Convert to PIL and plot
    img = F.to_pil_image(img)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Draw predicted boxes and masks
    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{labels[i]}:{scores[i]:.2f}",
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5))
        ax.imshow(masks[i], alpha=0.3, cmap='spring')

    plt.axis('off')
    plt.title(f"Prediction - Image Index: {index}")
    plt.savefig(os.path.join(save_dir, f"prediction_{index}.png"))
    plt.close()


# ==== Main Function ====
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load train and val datasets
    dataset = CocoInstanceDataset(
        img_folder=TrainConfig.train_images_dir,
        ann_file=TrainConfig.train_ann_file,
        transforms=get_transform(train=True)
    )
    dataset_val = CocoInstanceDataset(
        img_folder=TrainConfig.val_images_dir,
        ann_file=TrainConfig.val_ann_file,
        transforms=get_transform(train=False)
    )

    # Dataloaders
    data_loader = DataLoader(
        dataset, batch_size=TrainConfig.batch_size, shuffle=True,
        num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=utils.collate_fn
    )

    # Load model
    model = get_instance_segmentation_model(TrainConfig.num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=TrainConfig.lr,
                                momentum=TrainConfig.momentum,
                                weight_decay=TrainConfig.weight_decay)

    train_losses = []
    map_scores = []
    os.makedirs("results", exist_ok=True)

    # Training Loop
    for epoch in range(TrainConfig.num_epochs):
        print(f"\n Epoch {epoch + 1}/{TrainConfig.num_epochs}")
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        loss_avg = metric_logger.meters['loss'].global_avg
        train_losses.append(loss_avg)

        coco_eval = evaluate(model, data_loader_val, device=device)
        if coco_eval is not None and 'bbox' in coco_eval.coco_eval:
            map50 = coco_eval.coco_eval['bbox'].stats[1]  # mAP@0.5
            map_scores.append(map50)

    # Save final model
    torch.save(model.state_dict(), TrainConfig.save_model_path)
    print(f"\n Model saved to {TrainConfig.save_model_path}")

    # Plot Loss and mAP
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(map_scores, marker='o')
    plt.title("mAP@0.5 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5")

    plt.tight_layout()
    plt.savefig("results/metrics.png")
    plt.close()

    # Save sample prediction from validation set
    show_prediction(model, dataset_val, device, save_dir="results")


# ==== Entry Point ====
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
