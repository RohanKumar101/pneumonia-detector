from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow import keras
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Serve static frontend files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the trained model once at startup
model = keras.models.load_model("object.keras", compile=False)

labels = ['PNEUMONIA', 'NORMAL']

@app.get("/", response_class=HTMLResponse)
async def home():
    # Correct path that works both locally and on Render
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("L")  # convert to grayscale

    # Resize and normalize the image for prediction
    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = int((prediction > 0.5).astype("int32")[0][0])

    return JSONResponse({
        "prediction": labels[predicted_class],
        "confidence": float(prediction[0][0])
    })
