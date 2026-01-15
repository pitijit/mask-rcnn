class TrainConfig:
    # === Paths ===
    train_images_dir = "/dataset/images/train"
    train_ann_file = "/dataset/annotations/train.coco.json"
    val_images_dir = "/dataset/images/valid"
    val_ann_file = "/dataset/annotations/val.coco.json"
    save_model_path = "/models/maskrcnn_model.pth"

    # === Training Params ===
    num_classes = 7  # <-- include background!
    batch_size = 2
    num_epochs = 20
    lr = 0.001
    momentum = 0.9

    weight_decay = 0.0005
