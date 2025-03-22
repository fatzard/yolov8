from ultralytics import YOLO

# Load a model
model = YOLO("D:/Fatzard/desktop/YOLODetection/train/runs/detect/train2/weights/best.pt")  # 放训练好的模型参数

# Run batched inference on a list of images
results = model(["D:/Fatzard/desktop/YOLODetection/dataset/valid/images/68_jpg.rf.7e7688515ee35d70b5defdfc2ebbbbd6.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

