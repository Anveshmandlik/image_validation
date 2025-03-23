import traceback
import tempfile
import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Import validation functions from image_validation.py
from image_validation import (
    validate_dimensions,
    validate_background,
    validate_face_positioning,
    validate_emotion,
    validate_head_covering,
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    temp_path = None
    try:
        print(f"Received file: {file.filename}")

        contents = await file.read()
        if not contents:
            return JSONResponse(
                content={"error": "Empty file received"}, status_code=400
            )

        print(f"File size: {len(contents)} bytes")

        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                content={"error": "Could not decode image"}, status_code=400
            )

        print(f"Image decoded successfully. Shape: {image.shape}")

        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, image)
        print(f"Saved temporary file: {temp_path}")
        print("API Image Properties:")
        api_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"API Image mean brightness: {api_gray.mean()}")
        print(f"API Image brightness std: {api_gray.std()}")

        # Also check the temp file that's being used for head covering
        temp_image = cv2.imread(temp_path)
        temp_gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        print(f"Temp file mean brightness: {temp_gray.mean()}")
        print(f"Temp file brightness std: {temp_gray.std()}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("Starting validations...")
        dimensions_result, dimensions_message = validate_dimensions(image)
        print(f"Dimensions: {dimensions_result}, {dimensions_message}")

        background_result, background_message = validate_background(image)
        print(f"Background: {background_result}, {background_message}")

        face_positioning_result, face_positioning_message = validate_face_positioning(
            rgb_image
        )
        print(
            f"Face positioning: {face_positioning_result}, {face_positioning_message}"
        )

        emotion_result, emotion_message = validate_emotion(temp_path)
        print(f"Emotion: {emotion_result}, {emotion_message}")

        head_covering_result, head_covering_message = validate_head_covering(temp_path)
        print(f"Head covering: {head_covering_result}, {head_covering_message}")

        response = {
            "dimensions": {
                "valid": bool(dimensions_result),
                "message": dimensions_message,
            },
            "background": {
                "valid": bool(background_result),
                "message": background_message,
            },
            "face_positioning": {
                "valid": bool(face_positioning_result),
                "message": face_positioning_message,
            },
            "emotion": {"valid": bool(emotion_result), "message": emotion_message},
            "head_covering": {
                "valid": bool(head_covering_result),
                "message": head_covering_message,
            },
        }

        print("Validation completed successfully")
        return JSONResponse(content=response)

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error during validation: {error_msg}")
        print(error_trace)
        return JSONResponse(
            content={"error": f"Server error: {error_msg}"}, status_code=500
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Removed temporary file: {temp_path}")
            except Exception as e:
                print(f"Error removing temp file: {str(e)}")


# Serve HTML Frontend
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r", encoding="utf-8") as file:
        return file.read()


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
