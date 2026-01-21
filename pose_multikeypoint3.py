import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import time
from ultralytics.utils.plotting import Annotator


# Load YOLOv11 pose estimation model (equivalent to yolov8x-pose.pt)
pose_model = YOLO('yolo11n-pose_full_integer_quant_edgetpu.tflite', task='pose')

def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    return 360 - ang if ang > 180 else ang

class GesturedVideoCapture:
    is_recording: bool = False
#    video_writer: cv2.VideoWriter | None = None
    video_writer: cv2.VideoWriter
    capture_gesture_start_time: float = 0.0

    GESTURE_HOLD_TIME = 2.0 # 2 seconds

    def process_frame(self, frame, keypoints, current_time):
        nose = keypoints[0]
        nose_seen = nose[0] > 0 and nose[1] > 0
        left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
        right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        in_capture_gesture = (
            nose_seen and left_ear_seen and right_ear_seen and
            right_wrist[1] < nose[1] < left_wrist[1]
        )

        if self.is_recording and self.video_writer is not None:
            resized = cv2.resize(frame, (WRITER_WIDTH, WRITER_HEIGHT))
            self.video_writer.write(resized)

        if self.capture_gesture_start_time:
            if in_capture_gesture:
                if self.capture_gesture_start_time + self.GESTURE_HOLD_TIME < current_time:
                    # Hold for GESTURE_HOLD_TIME seconds - starting or stopping capture
                    if self.is_recording:
                        self.stop_and_save_capture()
                    else:
                        self.start_capture()

                    return
            else:
                self.capture_gesture_start_time = 0
        elif in_capture_gesture:
                self.capture_gesture_start_time = current_time


    def start_capture(self):
        print("Start capturing")
        self.is_recording = True
        self.capture_gesture_start_time = 0
        self.video_writer = cv2.VideoWriter(
            f"output_{int(time.time())}.mp4",
            cv2.VideoWriter_fourcc(*'avc1'),
            10,
            (WRITER_WIDTH, WRITER_HEIGHT)
        )


    def stop_and_save_capture(self):
        print("Stop capturing")
        self.is_recording = False
        self.capture_gesture_start_time = 0
        self.video_writer.release()
        self.video_writer = None


# Function to draw pose keypoints and skeleton (COCO 17-keypoint format)
def draw_angles(annotated_frame, keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if left_ear_seen and not right_ear_seen:
        angle_knee = get_angle(left_hip, left_knee, left_ankle)
        angle_hip = get_angle(left_shoulder, left_hip, left_knee)
    else:
        angle_knee = get_angle(right_hip, right_knee, right_ankle)
        angle_hip = get_angle(right_shoulder, right_hip, right_knee)

    knee_label_coordinates = [int(c) for c in left_knee]
    knee_label_coordinates[0] += 10
    knee_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_knee)}",
        knee_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

    hip_label_coordinates = [int(c) for c in left_hip]
    hip_label_coordinates[0] += 10
    hip_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_hip)}",
        hip_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

# Load input image
#image = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
new_width = 640  #4:3
new_height = 480
new_dimensions = (new_width, new_height)
prev_time = 0 

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)
    if not ret:
        break
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
 
    frames_per_last_second = 0
    video_capture = GesturedVideoCapture()

    results = pose_model(
        frame,
        conf=0.4,
        iou=0.45,
        classes=[0],  # Person class only
        device='tpu',
        half=True,
        verbose=False,
        imgsz=256
    )

    if not results:
           continue

    result = results[0]  #Mengambil hasil untuk frame saat ini
    annotator = Annotator(frame)  #Siapkan Annotator untuk frame ini
    all_keypoints_list = result.keypoints.xy.tolist()  # Untuk logika koordinat
    all_keypoints_data = result.keypoints.data         # Untuk menggambar kpts
    # Cek apakah ada orang yang terdeteksi
    if all_keypoints_list:
        for i in range(len(all_keypoints_list)):   #Loop melalui setiap orang yang terdeteksi
            keypoints_orang_ini = all_keypoints_list[i]
            if not keypoints_orang_ini:   #Pastikan data keypoints untuk orang ini tidak kosong
                continue
            annotator.kpts(all_keypoints_data[i], result.orig_shape, 5, True)  #Menggunakan data[i] untuk mendapatkan koordinat + confidence
            annotated_frame = annotator.result()  #Dapatkan frame yang sudah dianotasi sementara
            draw_angles(annotated_frame, keypoints_orang_ini)  #Hitung sudut dan proses data untuk orang ke-i
            video_capture.process_frame(annotated_frame, keypoints_orang_ini, time.time())  ## Jika video_capture perlu memproses tiap orang secara terpisah
    final_frame = annotator.result()

    cv2.putText(
        final_frame,
        f"FPS: {fps:.2f}",
        (10, 30),                    
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),                 
        2,
        cv2.LINE_AA
    )
    
    cv2.imshow("TEST", final_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
