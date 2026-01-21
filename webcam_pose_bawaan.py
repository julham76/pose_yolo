from ultralytics import YOLO
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model_s", "my_model.pt")
# MODEL_PATH = os.path.join(BASE_DIR, "model_s", "best_full_integer_quant_edgetpu.tflite")
model = YOLO('yolo11n-pose_full_integer_quant_edgetpu.tflite', task='pose')
#model = YOLO('/home/pi/segmen/my_coco128_full_integer_quant_edgetpu.tflite', task='segment')
#model = YOLO('/home/pi/segmen/yolo11sCOCO128_seg.pt', task='segment')
#model = YOLO('yolo11n-seg.pt', task='segment')

#cap = cv2.VideoCapture('https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/chunklist_w221624478.m3u8')
cap = cv2.VideoCapture('falldown.mp4')
new_width = 640  #4:3
new_height = 480
new_dimensions = (new_width, new_height)
prev_time = 0 

while True:
    ret, image = cap.read()
    frame = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    results = model.track(frame, persist=True, imgsz=256, conf=0.2, iou=0.45, half=True, verbose=False, device='tpu')
    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),                    
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),                 
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLOv11 Realtime Detection PPE", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
