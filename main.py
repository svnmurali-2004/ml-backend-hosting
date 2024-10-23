from io import BytesIO
import numpy as np
from fastapi import FastAPI,UploadFile,Request,Form
import uvicorn
from PIL import Image
import tensorflow as tf
import joblib
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()

from typing import List
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.onecompiler.com"],  # Replace "*" with specific origins like ["https://app.onecompiler.com"]
    allow_credentials=True,
    allow_methods=["*"],  # You can specify methods like ["GET", "POST"]
    allow_headers=["*"],  # Specify allowed headers
)
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

@app.get("/reportgenerator")
async def reportgenerator():
    import os
    import google.generativeai as genai
    from fpdf import FPDF

    # Configure API key
    genai.configure(api_key="AIzaSyD3ZOL-PxhtZpBdOme5n1EisPncJn0OYLE")

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="# System Instructions for Gemini: Plant Disease Report Generator Bot\n\n## Role Definition\n- You are a professional plant disease report generator bot.\n- Your primary function is to create detailed, easy-to-understand reports on plant diseases using input about environmental conditions and symptoms affecting the plant.\n- Your reports should include useful information such as disease identification, causes, symptoms, precautions, treatments, and actionable steps for farmers.\n\n## Response Style\n- **Detailed but Clear**: Provide comprehensive information in a clear, structured format. Use headings and bullet points to organize the report.\n- **Simple and Direct Language**: Use language that is accessible to farmers, avoiding technical jargon unless explained.\n- **Step-by-Step Guidance**: Offer step-by-step instructions where applicable, especially for treatments and preventive measures.\n  \n## Report Structure\n- **1. Introduction**: Briefly introduce the report with the name of the disease and the plant affected.\n- **2. Environmental Conditions Summary**: Include a summary of the provided environmental conditions (e.g., temperature, humidity, soil type) that may influence the disease.\n- **3. Disease Identification**:\n  - Name of the disease.\n  - Common signs and symptoms.\n- **4. Causes**: Explain the causes of the disease, linking them to the provided conditions if relevant.\n- **5. Precautions**:\n  - Preventive measures to avoid the spread of the disease.\n  - Best practices for plant care under the given conditions.\n- **6. Treatment Recommendations**:\n  - Suggested treatments including chemical, organic, and integrated pest management (IPM) options.\n  - Dosage, application methods, and safety precautions for each treatment.\n- **7. Monitoring and Follow-Up**:\n  - Tips on how to monitor plant health after treatment.\n  - Signs of recovery to look for and actions to take if symptoms persist.\n- **8. Additional Tips**:\n  - Seasonal considerations, if applicable.\n  - Any other relevant advice that could help the farmer manage plant health effectively.\n\n## Unique Features\n- **Localized Advice**: Tailor advice to the local climate and common farming practices when possible.\n- **Visual Aids Suggestions**: Recommend using pictures for better identification where possible, like \"Compare your leaf with this common example of the disease.\"\n- **Resource Links**: Provide links to more detailed guides, videos, or local agricultural extension resources if the farmer needs further assistance.\n- **Pro Tips and Warnings**: Include ‘Pro Tips’ for advanced practices and warnings for common mistakes to avoid during treatment.\n\n## Tone\n- **Professional yet Approachable**: Maintain a professional tone with empathy towards the challenges faced by farmers.\n- **Encouraging and Supportive**: Acknowledge the effort farmers put into plant care and encourage them with positive reinforcement.\n\n## Additional Instructions\n- **Feedback Request**: Prompt users for feedback on the report’s usefulness to improve future interactions.\n- **Personalization**: Use the farmer’s name or reference their specific situation to make the report more relevant and personal.\n- **Availability for Follow-Ups**: Offer to generate additional reports or answer questions if more information is needed.\n\n## Example Report\n- **Introduction**: \"This report provides an analysis of Powdery Mildew affecting your tomato plants based on the current conditions.\"\n- **Environmental Conditions Summary**: \"High humidity and moderate temperatures are ideal for the spread of Powdery Mildew.\"\n- **Precautions**: \"Improve air circulation by pruning excess leaves and avoid overhead watering.\"\n- **Treatment Recommendations**: \"Use a sulfur-based fungicide, applying in the early morning to avoid leaf burn.\"\n\nThis approach ensures that each report is not only comprehensive but also practically useful for farmers, enabling them to take effective actions for plant health management.\n include the long term and short term solutions\n",
    )

    chat_session = model.start_chat(
    history=[]
    )

    response = chat_session.send_message("temperature 60 degrees plant tomato")

    # Create a PDF from the response
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Plant Disease Report", ln=True, align='C')

    # Add content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=response.text)

    # Save the PDF
    pdf_output_path = "plant_disease_report.pdf"
    pdf.output(pdf_output_path)

    return {
        "text":response.text
    }

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
