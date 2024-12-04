import cv2
from ultralytics import YOLO
import os
import json
from tqdm import tqdm

# YOLO 모델 로드
model= YOLO('yolov8n.pt', task='detect')

# COCO 데이터셋 경로 설정
coco_images_path = 'C:/ultralytics/train2017'
coco_annotations_path = 'C:/ultralytics/annotations/person_keypoints_train2017.json'

# COCO 데이터셋 로드
with open(coco_annotations_path, 'r') as f:
    coco_data = json.load(f)

# 이미지 파일 리스트
image_files = [os.path.join(coco_images_path, img['file_name']) for img in coco_data['images']]


# 모델 학습
model.train(data='C:/Users/rainb/OneDrive/바탕 화면/final/data.yaml', epochs=50, batch=16)

print("모델 학습 완료")