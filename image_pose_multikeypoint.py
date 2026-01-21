import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLOv11 detection model (equivalent to yolov8x.pt)
det_model = YOLO('yolo11n_full_integer_quant_edgetpu.tflite', task='detect')

# Load YOLOv11 pose estimation model (equivalent to yolov8x-pose.pt)
pose_model = YOLO('yolo11n-pose_full_integer_quant_edgetpu.tflite', task='pose')

# Input and output image paths
image_path = 'tes.jpg'
save_path = 'tes_with_skeleton.jpg'

# Function to draw pose keypoints and skeleton (COCO 17-keypoint format)
def draw_pose(image, keypoints_xy, keypoints_conf, thickness=5):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image
    
    # COCO 17-keypoint skeleton (edges between keypoints)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for person_idx, (kpts, confs) in enumerate(zip(keypoints_xy, keypoints_conf)):
        kpts = kpts.cpu().numpy()  # Shape: (17, 2) [x, y]
        confs = confs.cpu().numpy()  # Shape: (17,) [confidence]
        
        # Draw keypoints
        for i, (x, y) in enumerate(kpts):
            if confs[i] > 0.5:  # Draw keypoints with sufficient confidence
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Draw skeleton lines
        for (start, end) in skeleton:
            if confs[start] > 0.5 and confs[end] > 0.5:
                start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, start_pt, end_pt, (255, 0, 0), thickness)
    
    return image

# Load input image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image ({image_path}).")
    exit()

# Perform YOLOv8 person detection
det_results = det_model(
    image,
    conf=0.25,
    iou=0.45,
    classes=[0],  # Person class only
    device='tpu',
    half=True,
    verbose=False
)

# Prepare for visualization
vis_image = image.copy()
person_bboxes = []

# Extract bounding boxes from detection results
for result in det_results:
    if result.boxes is not None and result.boxes.xyxy is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (xyxy)
        person_bboxes = boxes

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_image, "Person", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

# Perform pose estimation with YOLOv8-pose
pose_results = pose_model(
    image,
    conf=0.25,
    iou=0.45,
    classes=[0],  # Person class only
    device='tpu',
    half=True,
    verbose=False
)

# Extract keypoints and confidence scores
keypoints_xy = []
keypoints_conf = []
for result in pose_results:
    if result.keypoints is not None:
        keypoints_xy = result.keypoints.xy  # Shape: (num_persons, 17, 2) [x, y]
        keypoints_conf = result.keypoints.conf  # Shape: (num_persons, 17) [conf]

# Draw pose keypoints and skeleton
vis_image = draw_pose(vis_image, keypoints_xy, keypoints_conf)



# Save output image
cv2.imwrite(save_path, vis_image)
print(f"Output image saved to {save_path}")
