import logging
import os
import subprocess
import speech_recognition as sr
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio_wav(temp_wav_path="temp_patient_audio.wav", wait_for_speech=5):
    """
    Records audio from the microphone, saving it to a WAV file only if speech is detected.

    Args:
        temp_wav_path (str): Path to save the temporary WAV file.
        wait_for_speech (int): Max seconds to wait for user speech.
    
    Returns:
        bool: True if recording happened, False if no speech detected.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Preparing microphone... Hold on.")
            time.sleep(1)  # Warm up the mic before starting to listen
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            logging.info("Listening for your voice... Speak within 5 seconds or recording will stop.")

            try:
                audio_data = recognizer.listen(source, timeout=wait_for_speech, phrase_time_limit=10)
                with open(temp_wav_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
                logging.info(f"Audio recorded and saved to {temp_wav_path}")
                return True

            except sr.WaitTimeoutError:
                logging.warning("No speech detected within the wait time. Stopping.")
                return False

    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        return False

def convert_wav_to_mp3(input_wav_path, output_mp3_path):
    """
    Converts a WAV file to MP3 format using FFmpeg.

    Args:
        input_wav_path (str): Input WAV file path.
        output_mp3_path (str): Output MP3 file path.
    """
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", input_wav_path,
            "-vn",
            "-ar", "44100",
            "-ac", "1",
            "-b:a", "128k",
            output_mp3_path
        ]
        subprocess.run(command, check=True)
        logging.info(f"Converted {input_wav_path} to {output_mp3_path} using FFmpeg")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e}")

def transcribe_with_groq(stt_model, audio_filepath, groq_api_key):
    """
    Transcribes audio using Groq's Whisper model.

    Args:
        stt_model (str): Model name for STT.
        audio_filepath (str): Path to the audio file.
        groq_api_key (str): Groq API Key.

    Returns:
        str: Transcription text.
    """
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text
