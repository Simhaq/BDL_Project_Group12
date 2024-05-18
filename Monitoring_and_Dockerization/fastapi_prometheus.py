import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
import keras
import numpy as np
import sys
from PIL import Image
import io


import time

from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_client import disable_created_metrics

REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
counter = Counter('api_call_counter', 'number of times that API is called', ['endpoint', 'client'])
gauge = Gauge('api_runtime_secs', 'runtime of the method in seconds', ['endpoint', 'client'])

app = FastAPI()

# Function to Load the model
def load_model(path: str):
    return keras.models.load_model(path) 
# Function to format the image
def format_image(org_img):
    img_reshape=org_img.resize((32,32)) # Reshape the image to 32x32
    img_rgb=img_reshape.convert('RGB') # Convert image to RGB
    return img_rgb

# Function to Predict class
def predict_class(model, data_point):
    prediction = model.predict(data_point)
    class_number = np.argmax(prediction) # The index of the predicted class is the argmax of the output of the model
    class_dict={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',			 8:'ship',9:'truck'} #Dictionary that maps the predict class index with the class name
    predicted_class = class_dict[class_number]
    return predicted_class

# API endpoint
@REQUEST_DURATION.time()
@app.post('/predict')
async def predict(request:Request,file: UploadFile = File(...)):
    counter.labels(endpoint='/predict', client=request.client.host).inc()
    start = time.time()
    
    model_path = sys.argv[1]  # Get model path from command line argument
    loaded_model = load_model(model_path) # Load the model
    if file.content_type.startswith('image'): # Check if the input file is a image file
        image = await file.read() # Read the input file
        
    else:
        return {"error": "Uploaded file is not an image."}
    
    pil_image = Image.open(io.BytesIO(image)) # Open the image using PIL
    formatted_img=format_image(pil_image) # To resize the image to 32x32 and convert it to RGB
    numpy_image = np.array(formatted_img) # Convert PIL image to NumPy array
    numpy_image=numpy_image/255.0 # Convert the values to between 0 and 1
    data_point=numpy_image.reshape(1,32,32,3) # Reshape the image array such that it is compatible with model.predict
    predicted_class = predict_class(loaded_model, data_point) # Use the model to predict the digit
    
    
    time_taken = time.time() - start
    gauge.labels(endpoint='/predict', client=request.client.host).set(time_taken)
    return {"Predicted Class": predicted_class}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Specify the path of the model as a command line argument and no other arguments apart from that!!!")
        sys.exit(1)
    start_http_server(15000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
