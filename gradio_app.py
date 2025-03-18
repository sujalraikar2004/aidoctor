import os
import logging
import gradio as gr
from brain_of_the_doctor import encode_image, analyze_image_with_query, create_prescription
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts
from groq import Groq

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
STT_MODEL_ = "whisper-1"
TEXT_MODEL_ = "gpt-4-turbo"
VISION_MODEL_ = "gpt-4-turbo-vision"

# System prompt for text-based medical analysis
text_system_prompt = """Role: You are a professional medical assistant providing accurate and concise responses to patients' health queries.

Response Guidelines:

    Assessment:
        Provide a brief evaluation based on the symptoms and details shared by the patient.
        Use simple, patient-friendly language while maintaining medical accuracy.

    Medications:
        Suggest appropriate over-the-counter (OTC) or prescription medications.
        Include the medication names and their purpose (e.g., pain relief, infection treatment).
        Clearly specify if a doctor's prescription is required.

    Home Care Recommendations:
        Offer practical home remedies or lifestyle adjustments that can help alleviate symptoms.
        Ensure recommendations are evidence-based and safe for general use.

    Consultation Advice:
        Advise the patient if professional medical consultation is necessary.
        Specify urgency levels (e.g., immediate, within a few days, routine check-up).

Response Format:

    Keep the response under four sentences.
    Speak directly to the patient in a professional yet empathetic tone.
    Maintain clarity, avoiding complex medical jargon."""

def analyze_text_with_query(query):
    """Handles text-based medical queries using Groq API."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [
            {"role": "system", "content": text_system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Text analysis error: {str(e)}")
        return "Could not process query at this time."

# System prompt for image-based medical analysis
image_system_prompt = """You are a medical assistant. Analyze any visible skin conditions in the image and provide:
1. Brief professional assessment
2. Potential home care suggestions
3. Recommendation for professional consultation
Keep response under 3 sentences. Speak directly to the patient."""

def process_inputs(audio_path, text_query, image_path):
    """Processes user input (audio, text, or image) and generates medical analysis."""
    try:
        # Determine patient query
        patient_query = "No input provided"
        if audio_path:
            patient_query = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=audio_path,
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
            ) or "No audible input detected"
        elif text_query:
            patient_query = text_query
        
        # Determine processing method
        doctor_response = ""
        meaningful_query = patient_query not in ["No input provided", "No audible input detected"]
        
        if image_path:
            # Process image-based query
            doctor_response = analyze_image_with_query(
                query=f"{image_system_prompt}\nPatient says: {patient_query}",
                model="llama-3.2-11b-vision-preview",
                encoded_image=encode_image(image_path)
            ) or "Could not analyze image"
        elif meaningful_query:
            # Process text-based query
            doctor_response = analyze_text_with_query(f"Patient query: {patient_query}")
        else:
            doctor_response = "Please provide a medical query or upload an image for analysis."
        
        # Generate voice response
        audio_output = text_to_speech_with_gtts(
            input_text=doctor_response,
            output_filepath="doctor_response"
        ) if doctor_response else None
        
        # Generate prescription
        prescription_path = create_prescription(
            patient_query=patient_query,
            doctor_response=doctor_response
        ) if meaningful_query else None
        
        return [
            patient_query,
            doctor_response,
            audio_output if audio_output else None,
            prescription_path if prescription_path else None
        ]
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return [f"Error: {str(e)}"] * 4

# Gradio interface setup
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Voice Input"),
        gr.Textbox(label="Text Input", placeholder="Type your query here..."),
        gr.Image(type="filepath", label="Skin Condition Photo")
    ],
    outputs=[
        gr.Textbox(label="Patient Query"),
        gr.Textbox(label="Medical Analysis"),
        gr.Audio(label="Voice Response", autoplay=True),
        gr.File(label="Download Prescription")
    ],
    title="AI Smart Doctor Assistant",
    description="""Choose your input method (voice or text) and upload a skin photo.
    Get instant analysis with voice response and downloadable prescription.""",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_port=7860, show_error=True)
    print("Server started at http://localhost:7860")
