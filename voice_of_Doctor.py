from gtts import gTTS
from playsound import playsound
import os
import uuid

def speak_text_with_gtts(input_text: str, output_filepath: str = "doctor_response.mp3"):
    """
    Convert text to speech using gTTS and play the audio.

    Args:
        input_text (str): Text to convert into speech.
        output_filepath (str): Path to save the mp3 audio file.
    """
    # Generate a unique filename to avoid permission issues
    unique_filename = f"doctor_voice_{uuid.uuid4().hex[:8]}.mp3"
    
    try:
        tts = gTTS(text=input_text, lang='en', slow=False)
        tts.save(unique_filename)
        
        # Try to play the file
        try:
            playsound(unique_filename)
        except Exception as e:
            print(f"Warning: Could not play audio: {e}")
        
        # Copy the file to the requested location if different
        if output_filepath != unique_filename:
            try:
                # Make a copy of the file to the requested path
                with open(unique_filename, 'rb') as source_file:
                    content = source_file.read()
                    
                with open(output_filepath, 'wb') as target_file:
                    target_file.write(content)
            except Exception as e:
                print(f"Warning: Could not save to specified path '{output_filepath}': {e}")
                # Return the temp file path instead
                return unique_filename
                
        return output_filepath
    
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None
    finally:
        # Clean up the temp file after some time if it exists
        try:
            if os.path.exists(unique_filename) and unique_filename != output_filepath:
                # In a production app, you might want to schedule this for later
                # but for now we'll just try to remove it immediately
                try:
                    os.remove(unique_filename)
                except:
                    pass
        except:
            pass
