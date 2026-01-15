import os
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# === CONFIGURATION ===
IMAGE_DIR = "/images/val"
MODEL_PATH = "/maskrcnn_model.pth"
RESULTS_DIR = "/results"
NUM_CLASSES = 5
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results folder if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load Model ===
model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Inference Helper ===
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.pil_to_tensor(image).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    return image, prediction

# === Visualization and Save Helper ===
def save_result(image, prediction, save_path):
    image = np.array(image)
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    overlay = image.copy()

    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            mask = masks[i, 0] > 0.5
            color = np.array([255, 0, 0], dtype=np.uint8)  # Red
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)

    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = boxes[i]
            label = labels[i]
            score = scores[i]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=2, edgecolor='lime', facecolor='none'))
            ax.text(x1, y1 - 10, f"ID {label} {score:.2f}",
                    color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.7))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# === Run on All Images ===
if __name__ == "__main__":
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(IMAGE_DIR, filename)
            save_path = os.path.join(RESULTS_DIR, f"result_{filename}")
            print(f"Processing {filename}...")
            image, pred = predict(path)
            save_result(image, pred, save_path)

