import gradio as gr
from Brain_of_Bot import encode_image, analyze_image_with_query
from voice_of_patient import record_audio_wav, convert_wav_to_mp3, transcribe_with_groq
from voice_of_Doctor import speak_text_with_gtts
import os
import uuid

# Load environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
STT_MODEL = "whisper-large-v3"

def skintel_ai_pipeline(image, _):  # Underscore since audio input not used
    wav_path = f"temp_patient_audio_{uuid.uuid4().hex[:6]}.wav"
    mp3_path = f"patient_voice_{uuid.uuid4().hex[:6]}.mp3"
    output_voice_path = f"doctor_voice_{uuid.uuid4().hex[:6]}.mp3"

    # Step 1: Record
    recorded = record_audio_wav(temp_wav_path=wav_path, wait_for_speech=5)
    if not recorded:
        return "⚠️ No voice detected. Try again.", None

    # Step 2: Convert
    convert_wav_to_mp3(wav_path, mp3_path)

    # Step 3: Transcribe
    transcription_text = transcribe_with_groq(STT_MODEL, mp3_path, GROQ_API_KEY)
    if not transcription_text:
        return "⚠️ Transcription failed. Please retry.", None

    

    # Step 4: Image check
    if image is not None:
        encoded_image = encode_image(image)
        # Fixed function call - correct parameter order and names
        analysis_response = analyze_image_with_query(
            query=transcription_text,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            encoded_image=encoded_image
        )
    else:
        analysis_response = "No image provided. Please upload a skin image for analysis."

    # Step 5: Speak response
    voice_path = speak_text_with_gtts(analysis_response, output_voice_path)
    
    # Clean up temporary files
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")
    
    return analysis_response, voice_path

iface = gr.Interface(
    fn=skintel_ai_pipeline,
    inputs=[
        gr.Image(type="filepath", label="Upload Skin Image"),
        gr.Audio(sources=["microphone"], type="filepath", label="Voice Your Question")
    ],
    outputs=[
        gr.Textbox(label="Diagnosis / Response"),
        gr.Audio(label="Doctor Voice Response")
    ],
    title="🧠 Skintel AI - Skin Diagnosis Assistant",
    description="Upload a skin image and speak your query to get a diagnosis and voice feedback!"
)

if __name__ == "__main__":
    iface.launch()