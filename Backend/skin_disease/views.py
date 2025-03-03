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
import os
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage

# Configure Google API Key
genai.configure(
    api_key="AIzaSyBEjuyLDRRxkYef3KzBkbDO_xzEpDJMlTs",
    transport="rest"
)

# Load the trained disease prediction model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.h5")
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
    # Handle GET request - return the template
    if request.method == "GET":
        return render(request, "generate_report.html")
    
    # Handle POST request - process the image
    if request.method == "POST":
        if not request.FILES.get("image"):
            return JsonResponse({
                'success': False,
                'error': "No image file provided"
            }, status=400)

        image_file = request.FILES["image"]
        
        try:
            # Validate file type
            if not image_file.content_type.startswith('image/'):
                return JsonResponse({
                    'success': False,
                    'error': "File must be an image"
                }, status=400)

            # Preprocess the image
            try:
                image = Image.open(image_file).convert("RGB")
                image = image.resize((224, 224))  # Resize to model input size
                image_array = np.array(image) / 255.0  # Normalize
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f"Error preprocessing image: {str(e)}"
                }, status=400)

            try:
                # Predict the disease
                predictions = model.predict(image_array)
                predicted_class = disease_classes[np.argmax(predictions)]
                
                # Update the Gemini prompt to ensure structured response
                prompt = f"""
                Create a comprehensive medical analysis report for {predicted_class} using exactly this structure:

                1. DISEASE OVERVIEW:
                Basic Information:
                - Full name of the condition
                - Type of skin condition
                - Areas typically affected

                Common Symptoms:
                - Primary symptoms
                - Secondary symptoms
                - Visual characteristics

                Risk Factors:
                - Age groups most affected
                - Environmental factors
                - Genetic predisposition

                2. DISEASE STAGE ASSESSMENT:
                Early Stage Indicators:
                - Initial symptoms
                - Typical timeline
                - Early warning signs

                Progressive Stage Signs:
                - Advanced symptoms
                - Complications
                - Risk factors for progression

                3. TREATMENT RECOMMENDATIONS:
                Medications:
                - Over-the-counter options
                - Prescription medications
                - Topical treatments

                Medical Procedures:
                - Primary treatment options
                - Alternative therapies
                - Expected outcomes

                Treatment Timeline:
                - Initial treatment phase
                - Expected duration
                - Follow-up requirements

                4. HOME CARE PROTOCOL:
                Daily Skin Care:
                - Cleaning routine
                - Moisturizing recommendations
                - Sun protection measures

                Lifestyle Modifications:
                - Dietary recommendations
                - Activity restrictions
                - Environmental considerations

                5. MEDICAL CONSULTATION:
                Specialist Referral:
                - Type of specialist needed
                - When to seek immediate care
                - Required medical tests

                Follow-up Care:
                - Appointment frequency
                - Monitoring requirements
                - Progress indicators

                Please maintain this exact structure with main sections (1-5) and subsections. Use bullet points for all details.
                """

                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                response = gemini_model.generate_content(prompt)
                recommendations = response.text if hasattr(response, "text") else "No response generated."
                
                # Store in session
                request.session['predicted_class'] = predicted_class
                request.session['recommendations'] = recommendations

                return JsonResponse({
                    'success': True,
                    'predicted_class': predicted_class,
                    'recommendations': recommendations,
                    'disease_predictions': []
                })

            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f"Error during analysis: {str(e)}"
                }, status=500)

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }, status=500)

    return JsonResponse({
        'success': False,
        'error': "Invalid request method"
    }, status=405)

# Add a new endpoint for PDF download
@csrf_exempt
def download_pdf(request):
    if request.method == "POST":
        try:
            predicted_class = request.session.get('predicted_class', '')
            recommendations = request.session.get('recommendations', '')

            if not all([predicted_class, recommendations]):
                return JsonResponse({
                    'success': False,
                    'error': "No report data found. Please analyze an image first."
                }, status=400)

            # Create PDF buffer
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )

            # Enhanced Styles
            styles = getSampleStyleSheet()
            
            # Main Title
            styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1A76D1'),
                spaceAfter=30,
                alignment=1
            ))

            # Main Section Headers (1-5)
            styles.add(ParagraphStyle(
                name='MainSection',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.white,
                backColor=colors.HexColor('#1A76D1'),
                spaceBefore=15,
                spaceAfter=15,
                padding=10,
                alignment=0
            ))

            # Subsection Headers
            styles.add(ParagraphStyle(
                name='SubSection',
                parent=styles['Heading3'],
                fontSize=14,
                textColor=colors.HexColor('#1A76D1'),
                spaceBefore=12,
                spaceAfter=8,
                leftIndent=10
            ))

            # Content Text
            styles.add(ParagraphStyle(
                name='ContentText',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#333333'),
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=30
            ))

            # Build PDF content
            elements = []

            # Title and Header
            elements.append(Paragraph("Skin Disease Analysis Report", styles['ReportTitle']))

            # Info Box
            from datetime import datetime
            date_str = datetime.now().strftime("%B %d, %Y")
            ref_number = datetime.now().strftime("REF-%Y%m%d-%H%M%S")
            
            info_table = Table([
                [Paragraph(f"<b>Date:</b> {date_str}", styles['Normal']),
                 Paragraph(f"<b>Reference:</b> {ref_number}", styles['Normal'])],
                [Paragraph(f"<b>Diagnosed Condition:</b> {predicted_class}", styles['Normal']),
                 Paragraph("<b>Report Type:</b> Detailed Analysis", styles['Normal'])]
            ], colWidths=[3*inch, 3*inch])
            
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F7FF')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1A76D1')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1A76D1')),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 20))

            # Process recommendations into structured sections
            main_sections = {
                "1. DISEASE OVERVIEW": ["Basic Information", "Common Symptoms", "Risk Factors"],
                "2. DISEASE STAGE ASSESSMENT": ["Early Stage Indicators", "Progressive Stage Signs"],
                "3. TREATMENT RECOMMENDATIONS": ["Medications", "Medical Procedures", "Treatment Timeline"],
                "4. HOME CARE PROTOCOL": ["Daily Skin Care", "Lifestyle Modifications"],
                "5. MEDICAL CONSULTATION": ["Specialist Referral", "Follow-up Care"]
            }

            # Split content and process
            lines = recommendations.strip().split('\n')
            current_main_section = None
            current_sub_section = None
            content_buffer = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a main section
                if any(section in line for section in main_sections.keys()):
                    # Add previous section content if exists
                    if content_buffer:
                        for content in content_buffer:
                            elements.append(Paragraph(f"• {content}", styles['ContentText']))
                        content_buffer = []

                    current_main_section = line.strip()
                    elements.append(Paragraph(current_main_section, styles['MainSection']))
                    continue

                # Check if this is a subsection
                if current_main_section and any(sub in line for section in main_sections.values() for sub in section):
                    # Add previous subsection content if exists
                    if content_buffer:
                        for content in content_buffer:
                            elements.append(Paragraph(f"• {content}", styles['ContentText']))
                        content_buffer = []

                    current_sub_section = line.strip(':')
                    elements.append(Paragraph(current_sub_section, styles['SubSection']))
                    continue

                # Add content to buffer
                if line.startswith(('•', '-')):
                    line = line.lstrip('•- ').strip()
                if line:
                    content_buffer.append(line)

            # Add any remaining content
            if content_buffer:
                for content in content_buffer:
                    elements.append(Paragraph(f"• {content}", styles['ContentText']))

            # Footer
            elements.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'ReportFooter',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.gray,
                alignment=1
            )
            footer_text = """
                <para alignment="center">
                    This report is generated automatically using AI analysis.
                    Please consult with a healthcare professional for accurate diagnosis and treatment.
                    <br/><br/>
                    Generated by Skin Disease Analysis System
                    <br/>
                    Page 1
                </para>
            """
            elements.append(Paragraph(footer_text, footer_style))

            # Build PDF
            doc.build(elements)
            pdf_buffer.seek(0)

            response = HttpResponse(pdf_buffer, content_type="application/pdf")
            response["Content-Disposition"] = 'attachment; filename="skin_disease_report.pdf"'
            return response

        except Exception as e:
            return JsonResponse({
                'error': f"Error generating PDF: {str(e)}"
            }, status=500)

    return JsonResponse({
        'error': "Invalid request method"
    }, status=405)

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
            prompt = f"""
            You are a medical assistant. Based on the following conversation, provide helpful information about skin conditions.

            Previous Chat:
            {'\n'.join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history])}

            User: {user_message}

            Provide a clear and concise response about skin health.
            """

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

@csrf_exempt
def generate_pdf(request):
    if request.method == "POST":
        try:
            # Get predictions and recommendations from session
            disease_predictions = request.session.get('disease_predictions', [])
            predicted_class = request.session.get('predicted_class', '')
            recommendations = request.session.get('recommendations', '')

            # Generate PDF
            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

            margin = 50
            width, height = letter
            max_width = width - (2 * margin)
            y_position = height - 40

            # Title
            pdf.setFont("Helvetica", 16)
            pdf.drawCentredString(width / 2, y_position, "Skin Disease Report")
            y_position -= 25

            # Predicted Disease
            pdf.setFont("Helvetica", 12)
            pdf.drawString(margin, y_position, f"Predicted Disease: {predicted_class}")
            y_position -= 30

            # All Predictions
            pdf.setFont("Helvetica", 13)
            pdf.drawString(margin, y_position, "All Predictions:")
            y_position -= 20

            pdf.setFont("Helvetica", 11)
            for pred in disease_predictions:
                if y_position <= margin:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 11)
                    y_position = height - 50
                pdf.drawString(margin, y_position, 
                             f"{pred['disease']}: {pred['confidence']:.2f}%")
                y_position -= 14

            # Add recommendations section
            if recommendations:
                if y_position <= margin:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 11)
                    y_position = height - 50

                # Recommendations heading
                pdf.setFont("Helvetica", 13)
                pdf.drawString(margin, y_position, "Treatment and Prevention Recommendations:")
                y_position -= 20

                # Recommendations content
                pdf.setFont("Helvetica", 11)
                for line in recommendations.split('\n'):
                    if y_position <= margin:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50
                    wrapped_lines = simpleSplit(line, "Helvetica", 11, max_width)
                    for wrapped_line in wrapped_lines:
                        pdf.drawString(margin, y_position, wrapped_line)
                        y_position -= 14

            pdf.save()
            pdf_buffer.seek(0)

            response = HttpResponse(pdf_buffer, content_type="application/pdf")
            response["Content-Disposition"] = 'attachment; filename="skin_disease_report.pdf"'
            return response

        except Exception as e:
            return JsonResponse({"error": f"Error generating PDF: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)