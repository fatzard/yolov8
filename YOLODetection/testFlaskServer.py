import logging
import os

import aiofiles
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

logging.basicConfig(filename='server.log', level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # 允许所有域名跨域访问
# 加载YOLO模型
FireModel = YOLO("D:/Fatzard/desktop/YOLODetection/train/runs/detect/train2/weights/best.pt")


async def detect_objects_async(image, model, precision=0.3):
    """
    异步版本的目标检测函数。
    """
    results = model.predict(image)[0]
    # print(results)
    detections = []
    for i, box in enumerate(results.boxes.xyxy):
        conf = results.boxes.conf[i].item()
        if conf < precision:
            continue
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cls_id = int(results.boxes.cls[i])
        cls_name = results.names[cls_id]
        detection = {
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "orig_shape": results.boxes.orig_shape,
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        detections.append(detection)
    return detections


@app.route('/detect/<int:model_id>', methods=['POST'])
async def handle_request(model_id):
    print(model_id)
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify(error="No selected file"), 400

    # 确保 temp 目录存在
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)

    # 保存文件到临时路径
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    async with aiofiles.open(temp_path, 'rb') as f:
        file_bytes = await f.read()

    # 处理文件
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    logging.debug("Received image data: %s", image.shape)
    model = FireModel
    detections = await detect_objects_async(image, model)
    # 清理临时文件
    os.remove(temp_path)
    print(detections)
    return jsonify(detections)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9621)
