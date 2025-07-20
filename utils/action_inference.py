import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import subprocess
from concurrent.futures import ThreadPoolExecutor

def detect_action(input_path, output_path, skip_rate=3):
    # Initialize models
    model_yolo = YOLO('yolo11s-pose.pt')
    model = xgb.Booster()
    model.load_model('/home/robinpc/Desktop/FastApi_prac/action_recog_backend/models_file/new_tuned_trained_model.json')

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height}@{fps}fps")

    # Setup FFmpeg pipeline for direct H.264 encoding
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-movflags', 'faststart',  # Enable streaming
        output_path
    ]

    # Start FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Frame processing function
    def process_frame(frame, frame_count):
        annotated_frame = frame.copy()
        
        if frame_count % skip_rate == 0:
            results = model_yolo(frame, verbose=False)
            annotations = []
            
            for r in results:
                bound_boxes = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()

                for index, box in enumerate(bound_boxes):
                    if conf[index] > 0.55:
                        x1, y1, x2, y2 = box.tolist()
                
                        data = {}
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]
                        
                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        pred_probs = model.predict(dmatrix)
                        class_idx = int(np.argmax(pred_probs[0]))
                        class_label = {0: "Giving_Action", 1: "Money_counting", 2: "Normal"}[class_idx]
                        confidence = float(np.max(pred_probs[0]))
                        color = (0, 255, 0) if class_label == "Normal" else (0, 0, 255)
                        
                        annotations.append((int(x1), int(y1), int(x2), int(y2), class_label, confidence, color))
            return annotated_frame, annotations
        return None, None

    # Main processing loop
    frame_count = 0
    last_annotations = []
    
    try:
        with ThreadPoolExecutor() as executor:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                
                # Process frames in parallel
                if frame_count % skip_rate == 0:
                    annotated_frame, last_annotations = process_frame(frame, frame_count)
                else:
                    annotated_frame = frame.copy()
                
                # Apply last annotations
                for (x1, y1, x2, y2, class_label, confidence, color) in last_annotations:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cvzone.putTextRect(
                        annotated_frame,
                        f"{class_label} ({confidence:.2f})",
                        (x1, y1 + 50),
                        scale=3,
                        thickness=3
                    )
                
                # Write to FFmpeg stdin
                ffmpeg_process.stdin.write(annotated_frame.tobytes())
                
                if frame_count % 100 == 0:
                    print(f"Processed frame {frame_count}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
    finally:
        # Cleanup
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        if ffmpeg_process.returncode != 0:
            print(f"FFmpeg encoding failed with code {ffmpeg_process.returncode}")
        else:
            print(f"Successfully processed {frame_count} frames")
            print(f"Output saved to: {output_path}")

        # Verify output
        if os.path.exists(output_path):
            cap_out = cv2.VideoCapture(output_path)
            if cap_out.isOpened():
                print(f"Output verification: {int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
                cap_out.release()