import os
import logging
import gradio as gr
from brain_of_the_doctor import encode_image, analyze_image, create_prescription
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech
from groq import Groq
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Kannada": "kn"
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STT_MODEL = "whisper-large-v3"
TEXT_MODEL = "llama3-8b-8192"
VISION_MODEL = "llama-3.2-11b-vision-preview"

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
    Maintain clarity, avoiding complex medical jargon. esponses in {language} language. Use only {language} for responses. Follow these instructions"""

def analyze_text_with_query(query, language):
    """Handles text-based medical queries using Groq API."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [
            {"role": "system", "content": text_system_prompt.format(language=language)},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
            stream=True
        )
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
        return full_response
    except Exception as e:
        logging.error(f"Text analysis error: {str(e)}")
        return "Could not process query at this time."


# Modified image system prompt with language placeholder
image_system_prompt = """You are a medical assistant. Analyze the image and provide:
1. Brief professional assessment
2. Potential home care suggestions
3. Recommendation for professional consultation
Keep response under 3 sentences. Speak directly to the patient in {language} language. Use only {language} for responses. Follow these instructions."""

def process_inputs(audio_path, text_query, image_path, language):
    """Processes user inputs and generates medical analysis."""
    try:
        patient_query = "No input provided"
        if audio_path:
            patient_query = transcribe_with_groq(
                stt_model=STT_MODEL,
                audio_filepath=audio_path,
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                language=LANGUAGE_MAP[language]
            ) or "No audible input detected"
        elif text_query:
            patient_query = text_query
        
        doctor_response = ""
        meaningful_query = patient_query not in ["No input provided", "No audible input detected"]
        lang_code = LANGUAGE_MAP.get(language, "en")
        
        if image_path:
            # Format image prompt with selected language and encode image
            formatted_image_prompt = image_system_prompt.format(language=language)
            doctor_response = analyze_image(
                query=f"{formatted_image_prompt}\nPatient says: {patient_query}",
                image_path=image_path
            ) or "Could not analyze image"
        elif meaningful_query:
            doctor_response = analyze_text_with_query(
                query=f"Patient query: {patient_query}",
                language=language
            )
        else:
            doctor_response = "Please provide a medical query or upload an image for analysis."
        
        response_text = doctor_response
        audio_output = text_to_speech(text=response_text, language=lang_code) if doctor_response else None
        
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

# ... (keep the rest of the Gradio interface the same) ...
with gr.Blocks(title="AI Smart Medical Assistant") as demo:
    gr.Markdown("## AI Smart Medical Assistant")
    gr.Markdown("Supports English, Hindi, Marathi, Kannada. Upload an image for analysis.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
            text_input = gr.Textbox(label="Text Input", placeholder="Type your query here...")
            image_input = gr.Image(type="filepath", label="Skin Condition Photo")
            language_dropdown = gr.Dropdown(choices=["English", "Hindi", "Marathi", "Kannada"], 
                                          label="Select Language", value="English")
            submit_btn = gr.Button("Submit", variant="primary")
            
        with gr.Column():
            patient_query_box = gr.Textbox(label="Patient Query", interactive=False)
            analysis_box = gr.Textbox(label="Medical Analysis (Live)", elem_id="analysis_box")
            audio_output = gr.Audio(label="Voice Response", autoplay=True, interactive=False)
            prescription_output = gr.File(label="Download Prescription", interactive=False)

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, text_input, image_input, language_dropdown],
        outputs=[patient_query_box, analysis_box, audio_output, prescription_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)