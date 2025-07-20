from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import os
import shutil
import uuid
from pydantic import BaseModel
from action_recog_backend.utils.action_inference import detect_action


router = APIRouter()

UPLOAD_DIR = "/home/robinpc/Desktop/FastApi_prac/action_recog_backend/uploads"
OUTPUT_DIR = "/home/robinpc/Desktop/FastApi_prac/action_recog_backend/output"

processing_status = {}


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

processing_status = {}

@router.post("/run_inference")
async def run_inference(req: InferenceRequest, background_tasks: BackgroundTasks):
    input_path = os.path.join(UPLOAD_DIR, req.filename)
    output_path = os.path.join(OUTPUT_DIR, req.filename)

    if not os.path.exists(input_path):
        return {"error":" input file not found"}
    
    processing_status[req.filename] = True

    def process_and_update_status():
        detect_action(input_path, output_path)
        processing_status[req.filename] = False

    background_tasks.add_task(process_and_update_status)

    # Schedule the background task
    # background_tasks.add_task(detect_action, input_path, output_path, req.filename)
    
    # run infereence function that takes input_oath, output_path
    # detect_action(input_path, output_path)

    return {
        "status": "processing started",
        "filename": req.filename,
        # "output_video_url": f"/output/{req.filename}"
    }

@router.get("/check_status/{filename}")
async def check_status(filename: str):
    output_file = os.path.join(OUTPUT_DIR, filename)
    # if os.path.exists(output_file):
    #     return {"ready": True, "output_video_url": f"/output/{filename}"}
    # else:
    #     return {"ready": False}
    is_done = (not processing_status.get(filename, False)) and os.path.exists(output_file)
    return {"ready": is_done}

@router.get("/output/{filename}")
async def get_output_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path = file_path,
                        media_type = "video/mp4",
                        )


# from fastapi import Request, HTTPException
# from fastapi.responses import StreamingResponse
# import aiofiles

# @router.get("/output/{filename}")
# async def get_output_video(filename: str, request: Request):
#     file_path = os.path.join(OUTPUT_DIR, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     file_size = os.path.getsize(file_path)
#     headers = {}

#     range_header = request.headers.get('range')
#     if range_header:
#         # Example: "bytes=0-1023"
#         range_value = range_header.strip().split('=')[-1]
#         start_str, end_str = range_value.split('-')
#         start = int(start_str) if start_str else 0
#         end = int(end_str) if end_str else file_size - 1
#         length = end - start + 1

#         async def iter_file():
#             async with aiofiles.open(file_path, 'rb') as f:
#                 await f.seek(start)
#                 remaining = length
#                 while remaining > 0:
#                     chunk_size = min(4096, remaining)
#                     chunk = await f.read(chunk_size)
#                     if not chunk:
#                         break
#                     remaining -= len(chunk)
#                     yield chunk

#         headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
#         headers['Accept-Ranges'] = 'bytes'
#         headers['Content-Length'] = str(length)

#         return StreamingResponse(iter_file(), status_code=206, headers=headers, media_type="video/mp4")

#     else:
#         # no Range header — send full file
#         async def iter_file():
#             async with aiofiles.open(file_path, 'rb') as f:
#                 while True:
#                     chunk = await f.read(4096)
#                     if not chunk:
#                         break
#                     yield chunk
#         headers['Content-Length'] = str(file_size)
#         return StreamingResponse(iter_file(), headers=headers, media_type="video/mp4")
