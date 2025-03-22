import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.pt")
    # 训练模型
    results = model.train(data="D:\Fatzard\desktop\YOLODetection\dataset\data.yaml",
                          epochs=10,
                          patience=30,
                          batch=4,
                          num_workers=1,
                          device=torch.device('cuda'),
                          amp=False)
