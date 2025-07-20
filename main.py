from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from action_recog_backend.api import routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],

)

app.include_router(routes.router)

# Serve the oputput folde for your annotated videos
app.mount("/output", StaticFiles(directory="/home/robinpc/Desktop/FastApi_prac/action_recog_backend/output"),name = "output")