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
                
                # Generate treatment plan using Gemini API
                prompt = (
                    f"Provide a comprehensive analysis for {predicted_class} with the following sections:\n\n"
                    f"1. Disease Overview:\n"
                    f"- Brief description of the condition\n"
                    f"- Common symptoms\n"
                    f"- Typical causes\n\n"
                    f"2. Disease Stage Assessment:\n"
                    f"- Possible stages of this condition\n"
                    f"- General characteristics of early stages (Stage 1-2)\n"
                    f"- Warning signs for advanced stages\n\n"
                    f"3. Treatment Options:\n"
                    f"- Recommended medications (for early stages)\n"
                    f"- Treatment procedures\n"
                    f"- Estimated treatment duration\n\n"
                    f"4. Medical Consultation:\n"
                    f"- Type of specialists to consult\n"
                    f"- When to seek immediate medical attention\n\n"
                    f"5. Home Care and Prevention:\n"
                    f"- Self-care measures\n"
                    f"- Lifestyle modifications\n"
                    f"- Prevention strategies\n\n"
                    f"Format with clear headings and bullet points. Be specific about medications and specialists."
                )

                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                response = gemini_model.generate_content(prompt)
                recommendations = response.text if hasattr(response, "text") else "No response generated."
                
                # Remove Markdown-style formatting
                cleaned_recommendations = re.sub(r"\*\*(.*?)\*\*", r"\1", recommendations)

                # Store in session
                request.session['predicted_class'] = predicted_class
                request.session['recommendations'] = cleaned_recommendations

                # Return JSON response with empty array for disease_predictions to prevent frontend errors
                return JsonResponse({
                    'success': True,
                    'predicted_class': predicted_class,
                    'recommendations': cleaned_recommendations,
                    'disease_predictions': []  # Add this empty array to prevent frontend errors
                })

            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f"Error during prediction: {str(e)}"
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
            styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1A76D1'),
                spaceAfter=30,
                alignment=1
            ))
            
            styles.add(ParagraphStyle(
                name='ReportSection',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.white,
                backColor=colors.HexColor('#1A76D1'),
                spaceBefore=15,
                spaceAfter=15,
                padding=10,
                borderRadius=5,
                alignment=0
            ))

            styles.add(ParagraphStyle(
                name='ReportSubHeader',
                parent=styles['Heading3'],
                fontSize=14,
                textColor=colors.HexColor('#1A76D1'),
                spaceBefore=15,
                spaceAfter=10,
                borderBottom=1,
                borderColor=colors.HexColor('#1A76D1')
            ))

            styles.add(ParagraphStyle(
                name='ReportContent',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#333333'),
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=20
            ))

            # Add new styles for boxes and highlights
            styles.add(ParagraphStyle(
                name='WarningText',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#CC0000'),
                backColor=colors.HexColor('#FFE8E8'),
                borderColor=colors.HexColor('#CC0000'),
                borderWidth=1,
                borderPadding=8,
                spaceBefore=10,
                spaceAfter=10
            ))

            # Build PDF content
            elements = []

            # Title and Date
            elements.append(Paragraph("Skin Disease Analysis Report", styles['ReportTitle']))
            
            # Date and Reference
            from datetime import datetime
            date_str = datetime.now().strftime("%B %d, %Y")
            ref_number = datetime.now().strftime("REF-%Y%m%d-%H%M%S")
            
            info_data = [
                [Paragraph(f"<b>Date:</b> {date_str}", styles['Normal']),
                 Paragraph(f"<b>Reference:</b> {ref_number}", styles['Normal'])]
            ]
            info_table = Table(info_data, colWidths=[3*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F7FF')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1A76D1')),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 20))

            # Diagnosis Box
            diagnosis_table = Table([
                [Paragraph("<b>DIAGNOSIS</b>", styles['ReportSection'])],
                [Paragraph(f"<b>Identified Condition:</b> {predicted_class}", 
                          ParagraphStyle(
                              'DiagnosisText',
                              parent=styles['Normal'],
                              fontSize=14,
                              textColor=colors.HexColor('#333333'),
                              spaceBefore=10,
                              spaceAfter=10,
                              leftIndent=10
                          ))]
            ], colWidths=[7*inch])
            
            diagnosis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F7FF')),
                ('BOX', (0, 1), (-1, -1), 1, colors.HexColor('#1A76D1')),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(diagnosis_table)
            elements.append(Spacer(1, 20))

            # Add Disease Information Section
            elements.append(Paragraph("DISEASE INFORMATION", styles['ReportSection']))
            
            # Process disease overview
            sections = recommendations.split('\n\n')
            current_section = ""
            
            for section in sections:
                if ':' in section.split('\n')[0]:
                    heading, content = section.split(':', 1)
                    heading = heading.strip()
                    
                    # Style different sections appropriately
                    if "Disease Overview" in heading:
                        elements.append(Paragraph(heading, styles['ReportSubHeader']))
                        elements.append(Paragraph(content.strip(), styles['ReportContent']))
                        elements.append(Spacer(1, 10))
                    
                    elif "Disease Stage" in heading:
                        elements.append(Paragraph(heading, styles['ReportSubHeader']))
                        stage_content = content.strip()
                        elements.append(Paragraph(stage_content, styles['ReportContent']))
                        
                        # Add warning box for advanced stages
                        warning_text = (
                            "<b>Important Notice:</b> If you notice signs of advanced stages, "
                            "please seek immediate medical attention."
                        )
                        elements.append(Paragraph(warning_text, styles['WarningText']))
                        elements.append(Spacer(1, 10))
                    
                    elif "Treatment Options" in heading:
                        elements.append(Paragraph(heading, styles['ReportSubHeader']))
                        
                        # Create a treatment table
                        treatment_lines = content.strip().split('\n')
                        treatment_data = [[Paragraph("<b>Treatment Type</b>", styles['ReportContent']), 
                                         Paragraph("<b>Details</b>", styles['ReportContent'])]]
                        
                        for line in treatment_lines:
                            if line.strip():
                                if ':' in line:
                                    type_, detail = line.split(':', 1)
                                    treatment_data.append([
                                        Paragraph(type_.strip(), styles['ReportContent']),
                                        Paragraph(detail.strip(), styles['ReportContent'])
                                    ])
                        
                        treatment_table = Table(treatment_data, colWidths=[2*inch, 5*inch])
                        treatment_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F7FF')),
                            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1A76D1')),
                            ('PADDING', (0, 0), (-1, -1), 6),
                        ]))
                        elements.append(treatment_table)
                        elements.append(Spacer(1, 10))
                    
                    elif "Medical Consultation" in heading:
                        elements.append(Paragraph(heading, styles['ReportSubHeader']))
                        
                        # Format specialist information
                        specialist_info = content.strip().split('\n')
                        for info in specialist_info:
                            if info.strip():
                                elements.append(Paragraph(f"• {info.strip()}", styles['ReportContent']))
                        elements.append(Spacer(1, 10))
                    
                    elif "Home Care and Prevention" in heading:
                        elements.append(Paragraph(heading, styles['ReportSubHeader']))
                        care_lines = content.strip().split('\n')
                        for line in care_lines:
                            if line.strip():
                                elements.append(Paragraph(f"• {line.strip()}", styles['ReportContent']))
                        elements.append(Spacer(1, 15))

            # Add a Quick Reference Box
            quick_ref_data = [
                [Paragraph("<b>QUICK REFERENCE</b>", styles['ReportSection'])],
                [Paragraph(
                    "<b>Key Actions:</b><br/>"
                    "1. Consult recommended specialists<br/>"
                    "2. Follow prescribed medications<br/>"
                    "3. Implement prevention measures<br/>"
                    "4. Schedule regular check-ups",
                    styles['ReportContent']
                )]
            ]
            quick_ref_table = Table(quick_ref_data, colWidths=[7*inch])
            quick_ref_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F7FF')),
                ('BOX', (0, 1), (-1, -1), 1, colors.HexColor('#1A76D1')),
                ('PADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(quick_ref_table)

            # Footer with disclaimer
            elements.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.gray,
                alignment=1
            )
            footer_text = """
                This report is generated automatically using AI analysis. 
                Please consult with a healthcare professional for accurate diagnosis and treatment.
                <br/>
                Generated by Skin Disease Analysis System
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