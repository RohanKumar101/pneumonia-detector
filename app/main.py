from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow import keras
from io import BytesIO
from PIL import Image

app = FastAPI()

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model once at startup
model = keras.models.load_model("object.keras", compile=False)


labels = ['PNEUMONIA', 'NORMAL']

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("app/static/index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("L")  # grayscale

    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))

    prediction = model.predict(img_array)
    predicted_class = (prediction > 0.5).astype("int32")[0][0]

    return JSONResponse({
        "prediction": labels[predicted_class],
        "confidence": float(prediction)
    })
