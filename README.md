# Image Validation System

A web-based tool that validates passport photos against standard requirements through automated image analysis.

## Project Overview

This project aims to help users verify that their passport photos meet official requirements before submission. The automated validator checks for common compliance issues to help avoid photo rejections in passport applications.

The system analyzes uploaded photos for:
- Correct dimensions (413x531 pixels)
- White background
- Proper face positioning
- Neutral expression
- Absence of head coverings

## Tech Stack

- **Backend**: Python with FastAPI
- **Computer Vision**: OpenCV, MTCNN, and DeepFace
- **Frontend**: HTML, CSS, and JavaScript
- **Development Environment**: VS Code

## Installation & Setup

1. **Clone the repository**
   ```
   git clone https://github.com/Anveshmandlik/vfs_project.git
   cd vfs_project
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   # Avtivate the environment on windows: venv\Scripts\activate
   ```

3. **Install requirements**
   ```
   pip install fastapi uvicorn python-multipart opencv-python numpy mtcnn deepface
   ```

4. **Create the project structure**
   ```
   mkdir static
   # Place index.html in the static folder
   ```

## Running the Application

1. **Start the FastAPI server (Run this in the terminal)**
   ```
   uvicorn api:app --port 8001
   ```

2. **Access the web interface**
   Open your browser and go to http://127.0.0.1:8001/

3. **Upload and validate photos**
   - Click "Choose File" to select a passport photo
   - Click "Validate" to analyze the photo
   - Review the validation results

## How It Works

1. User uploads an image through the web interface
2. The backend processes the image using validation techniques:
   - Dimension validation compares image size against requirements
   - Background validation analyzes pixel colors
   - Face positioning uses MTCNN to detect facial features
   - Expression analysis uses DeepFace to detect emotions
   - Head covering detection looks for obstructions in the upper face area
3. Results are returned to the frontend and displayed to the user

## Challenges & Solutions

The development process included several technical challenges:

- **Integration issues**: Connecting the front-end with the API required careful error handling
- **DeepFace compatibility**: The emotion validation needed modifications to handle in-memory images
- **Error handling**: Robust error handling was implemented for all validation steps

## Future Improvements

Planned future enhancements include:

- Displaying the image while the user uploads it on the interface
- Detecting the accessories on face (ex: Glasses)
- Saving validation history
