import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO   # Replace with correct import for YOLOv11 if needed
import shutil

app = FastAPI()

MODEL_FOLDER = "models"
MODEL_PATH = next((os.path.join(MODEL_FOLDER, f) for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')), None)
model = YOLO(MODEL_PATH)

@app.post("/pose/image")
async def pose_from_image(file: UploadFile = File(...)):
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(temp_path)
   
    results = model(img)
    poses = []
    for kp in results[0].keypoints:
        flat_kp = kp.data.cpu().numpy().flatten().astype(int).tolist()
        poses.append(flat_kp)
    os.remove(temp_path)
    return JSONResponse(content=poses)

@app.post("/pose/video")
async def pose_from_video(file: UploadFile = File(...)):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    cap = cv2.VideoCapture(temp_path)
    all_poses = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for kp in results[0].keypoints.cpu().numpy():
            flat_kp = kp.flatten().astype(int).tolist()
            all_poses.append(flat_kp)
        break  # Remove this break to process all frames
    cap.release()
    os.remove(temp_path)
    return JSONResponse(content=all_poses)

@app.post("/pose/rtsp")
async def pose_from_rtsp(rtsp_url: str = Form(...)):
    cap = cv2.VideoCapture(rtsp_url)
    all_poses = []
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        for kp in results[0].keypoints.cpu().numpy():
            flat_kp = kp.flatten().astype(int).tolist()
            all_poses.append(flat_kp)
    cap.release()
    return JSONResponse(content=all_poses)

@app.post("/pose/webcam")
async def pose_from_webcam():
    cap = cv2.VideoCapture(0)
    all_poses = []
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        for kp in results[0].keypoints.cpu().numpy():
            flat_kp = kp.flatten().astype(int).tolist()
            all_poses.append(flat_kp)
    cap.release()
    return JSONResponse(content=all_poses)
