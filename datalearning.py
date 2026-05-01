from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 윈도우 멀티프로세싱 지원을 위해 필요
    multiprocessing.freeze_support()

    # 1. YOLOv11 나노(nano) 모델 로드
    model = YOLO("yolo11n.pt") 

    # 2. 데이터셋 학습 
    # workers=0 으로 설정하면 경로 문제를 피하고 안정적으로 학습을 시작할 수 있습니다.
    model.train(
        data="C:/Project/data.yaml", 
        epochs=50, 
        imgsz=640, 
        workers=0  # 파일 경로 에러가 날 때 0으로 설정하면 도움이 됩니다.
    )