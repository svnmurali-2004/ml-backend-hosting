from io import BytesIO
import numpy as np
from fastapi import FastAPI,UploadFile,Request,Form
import uvicorn
from PIL import Image
import tensorflow as tf
import joblib
import pandas as pd
import json
app=FastAPI()

from typing import List

# @app.post("/aqiprediction")
# async def aqiprediction(request:Request):
#     try:
#         body=await request.body()
#
#         data=pd.Series(body.data)
@app.post("/potatodiseaseprediction")
async def potatodiseasepredicttion(file:UploadFile):
    class_prediction=["early_binding","late_binding","healthy"]
    image=Image.open(BytesIO(await file.read()))
    image=np.array(image)
    image=np.expand_dims(image,axis=0)
    model=tf.keras.models.load_model("./models/plantvillage.keras")
    res=model.predict(image)[0]
    # index=np.argmax(res)[0]
    # print(index,"index")
    print(res)
    print(np.argmax(res))
    index=np.argmax(res)
    print(res[index])
    return {
        "predictedclass":class_prediction[index],
        "conf":"%.4f"%float(res[index])

    }
@app.post("/irispredictor")
async def irispredictor(SepalLengthCm: float = Form(...),
    SepalWidthCm: float = Form(...),
    PetalLengthCm: float = Form(...),
    PetalWidthCm: float = Form(...)):
    data = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
    labellist=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    df=pd.DataFrame(data,columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    model=joblib.load("./models/Irismodel.joblib")
    res=model.predict(df)
    return {
        "ok":True,
        "class":labellist[res[0]]
    }
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
