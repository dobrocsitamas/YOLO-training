import sys
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, r'C:\Users\admin\Trafic_mojo_2\TM_modulok_py\Traffic_Mojo_2_0')

model_path = r'C:\Users\admin\Trafic_mojo_2\TM_modulok_py\Traffic_Mojo_2_0\YOLO_models\Traffic14_V3\traffic14_v3.pt'

print("Loading model...")
model = YOLO(model_path)

test_vid = r'C:\Users\admin\Trafic_mojo_2\Munkaközi\Traffic_mojo_2_0_bemutató.mp4'
if os.path.exists(test_vid):
    print(f"\nUsing video: {test_vid}")
    cap = cv2.VideoCapture(test_vid)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")
    
    # Try multiple frames: 10%, 30%, 50%, 70%, 90%
    for pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
        frame_idx = int(frame_count * pct)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            print(f"\n--- Inference on Frame {frame_idx} ({int(pct*100)}%) ---")
            results = model(frame, imgsz=640, conf=0.1, verbose=False) # Reduced confidence
            if results and results[0].boxes:
                boxes = results[0].boxes
                print(f"Total detections: {len(boxes)}")
                # Show top 5 detections
                for i in range(min(5, len(boxes))):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    print(f"  {i}: {cls_name} ({conf:.3f})")
            else:
                print("No detections.")
        else:
            print(f"Failed to read frame {frame_idx}")
    cap.release()
else:
    print(f"Video not found: {test_vid}")
