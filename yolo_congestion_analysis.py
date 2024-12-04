import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# YOLOv11 모델 로드
model = YOLO('yolov11s.pt')  # YOLOv11n 모델을 사용합니다.

def analyze_congestion(image_path, density_thresholds=(5, 15)):
    """
    이미지를 받아 사람 객체를 탐지하고 혼잡 상태를 분석하여 표시합니다.

    Args:
        image_path (str): 분석할 이미지 경로
        density_thresholds (tuple): (원활, 경고) 기준이 되는 사람 수 임계값

    Returns:
        None
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO 모델을 통해 객체 탐지 수행
    results = model(img_rgb)

    # 사람 객체만 필터링 (클래스 ID: 0)
    persons = [result for result in results[0].boxes if result.cls == 0]
    person_count = len(persons)

    # 밀집도 상태 계산
    if person_count < density_thresholds[0]:
        status = "원활"
    elif person_count < density_thresholds[1]:
        status = "경고"
    else:
        status = "혼잡"

    # 탐지된 객체를 바운딩 박스로 표시
    for person in persons:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 상태 텍스트 추가
    cv2.putText(img_rgb, f"Status: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 결과 이미지 표시
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# 테스트 실행
if __name__ == "__main__":
    # 처리할 이미지 경로
    image_file = "example.jpg"  # 여기에 이미지 경로를 입력하세요
    analyze_congestion(image_file)
