from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import os
import shutil
from pydantic import BaseModel
from action_recog_backend.utils.optimized_action_inference import detect_action


router = APIRouter()

UPLOAD_DIR = "/home/robinpc/Desktop/FastApi_prac/action_recog_backend/uploads"
OUTPUT_DIR = "/home/robinpc/Desktop/FastApi_prac/action_recog_backend/output"

os.makedirs(UPLOAD_DIR, exist_ok = True)
os.makedirs(OUTPUT_DIR, exist_ok = True)

@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open (file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    return {"filename": file.filename}


class InferenceRequest(BaseModel):
    filename: str

@router.post("/run_inference")
async def run_inference(req: InferenceRequest, background_tasks: BackgroundTasks):
    input_path = os.path.join(UPLOAD_DIR, req.filename)
    output_path = os.path.join(OUTPUT_DIR, req.filename)

    if not os.path.exists(input_path):
        return {"error":" input file not found"}
    
    # Schedule the background task
    background_tasks.add_task(detect_action, input_path, output_path)
    
    # run infereence function that takes input_oath, output_path
    # detect_action(input_path, output_path)

    return {
        "status": "processing started",
        "filename": req.filename,
        "output_video_url": f"/output/{req.filename}"
    }

@router.get("/output/{filename}")
async def get_output_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path = file_path, media_type = "video/mp4", filename=filename)

