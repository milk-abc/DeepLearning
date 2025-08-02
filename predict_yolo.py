import argparse
import json
import os
from ultralytics import YOLO

def predict_yolo(image_path, model_path='runs/detect/train4/weights/best.pt', output_json='result.json'):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"[❌] 找不到图片：{image_path}")
        return

    if not os.path.exists(model_path):
        print(f"[❌] 找不到模型：{model_path}")
        return

    # 加载模型
    model = YOLO(model_path)

    # 推理
    results = model(image_path)

    # 解析结果
    output_data = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            output_data.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

    # 保存 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 预测完成，已保存为 {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 户型图目标检测并输出 JSON")
    parser.add_argument('--image', required=True, help="要预测的图片路径")
    parser.add_argument('--model', default='runs/detect/train4/weights/best.pt', help="YOLOv8 模型路径")
    parser.add_argument('--output', default='result.json', help="输出 JSON 文件名")

    args = parser.parse_args()

    predict_yolo(args.image, args.model, args.output)