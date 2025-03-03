from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import re

# Configure Google API Key
genai.configure(api_key="AIzaSyBEjuyLDRRxkYef3KzBkbDO_xzEpDJMlTs")

# Load the trained disease prediction model
MODEL_PATH = "C:\\Users\\HARIPRASATH\\Downloads\\HC\\healthcare\\model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# List of disease classes
disease_classes=['Actinic Keratoses',
   'Basal Cell Carcinoma',
   'Benign Keratosis',
   'Dermatofibroma',
   'Melanoma',
   'Melanocytic Nevi',
   'Vascular naevus'
]
@csrf_exempt
def generate_skin_report(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        try:
            # Preprocess the image
            image = Image.open(image_file).convert("RGB")
            image = image.resize((224, 224))  # Resize to model input size
            image_array = np.array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Predict the disease
            predictions = model.predict(image_array)
            predicted_class = disease_classes[np.argmax(predictions)]
            
            # Generate treatment plan using Gemini API
            prompt = (
                f"Provide a concise and meaningful treatment plan for {predicted_class} within a single page. "
                f"Include: an image of the affected area, the disease name, and a well-structured list of treatments, remedies, and precautions."
            )

            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)
            generated_text = response.text if hasattr(response, "text") else "No response generated."
            
            # Remove Markdown-style bold formatting (**bold text**)
            cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", generated_text)

            # Generate PDF
            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

            margin = 50  # Reduced margin
            width, height = letter
            max_width = width - (2 * margin)
            y_position = height - 40  # Reduced initial space at the top

            # Title
            pdf.setFont("Helvetica", 16)
            pdf.drawCentredString(width / 2, y_position, "Skin Disease Report")
            y_position -= 25  # Reduced spacing below the title

            # Predicted Disease
            pdf.setFont("Helvetica", 12)
            pdf.drawString(margin, y_position, f"Predicted Disease: {predicted_class}")
            y_position -= 30  # Adjusted spacing below the disease name

            # Treatment Plan Heading
            pdf.setFont("Helvetica", 13)
            pdf.drawString(margin, y_position, "Treatment Plan:")
            y_position -= 20  # Slightly reduced spacing

            # Content
            pdf.setFont("Helvetica", 11)
            for line in cleaned_text.split("\n"):
                wrapped_lines = simpleSplit(line, "Helvetica", 11, max_width)
                for wrapped_line in wrapped_lines:
                    if y_position <= margin:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50  # Adjusted to fit more text
                    pdf.drawString(margin, y_position, wrapped_line)
                    y_position -= 14  # Reduced line spacing

            pdf.save()
            pdf_buffer.seek(0)

            # Return PDF as response
            response = HttpResponse(pdf_buffer, content_type="application/pdf")
            response["Content-Disposition"] = 'attachment; filename="skin_disease_report.pdf"'
            return response

        except Exception as e:
            return JsonResponse({"error": f"Error processing image: {str(e)}"}, status=500)

    return render(request, "generate_report.html")


def myself(request):
    return render(request,'index.html')

import google.generativeai as genai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

API_KEY = "AIzaSyBEjuyLDRRxkYef3KzBkbDO_xzEpDJMlTs"  # Replace with a secure API key
genai.configure(api_key=API_KEY)

# Store chat history in session
@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            # Retrieve previous chat history from session
            chat_history = request.session.get("chat_history", [])

            # Gemini Prompt to ensure remedies or clarifications
            prompt = (
                f"Previous Chat:\n{chat_history}\n\n"
                f"User: {user_message}\n"
                "Respond concisely in 10 words or fewer. "
                "Do not ask questions. Only provide direct and meaningful answers."
            )




            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Update chat history
            chat_history.append({"user": user_message, "bot": response_text})
            request.session["chat_history"] = chat_history

            return JsonResponse({"response": response_text, "chat_history": chat_history})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid request"}, status=400)



def page(request):
    return render(request,'skin_disease/404.html')