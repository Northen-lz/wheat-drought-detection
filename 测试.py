from ultralytics import YOLO
import multiprocessing

def train():
    model = YOLO("D:/pyhon/xiangmu/yolov8n.pt")

    model.train(
        data="D:/pyhon/xiangmu/xiaomai/wheat500/wheat500.yaml",
        epochs=50,
        imgsz=512,
        batch=8,
        device="0",
        workers=2,
        project="D:/pyhon/xiangmu/detect_compare_runs",
        name="yolov8n",
        exist_ok=True
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()