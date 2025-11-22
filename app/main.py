from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.validators import check_image_corruption


app = FastAPI(
    title="Image Integrity API",
    description="Advanced corruption + security validation API",
    version="1.0.0"
)

# CORS for React Native / Web Apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    results = check_image_corruption(file_bytes)
    return results
