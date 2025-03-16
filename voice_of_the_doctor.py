from gtts import gTTS
from pydub import AudioSegment
import os
import platform
import subprocess
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def text_to_speech_with_gtts(
    input_text: str,
    output_filepath: str = "doctor_response",
    lang: str ='en',
    slow: bool = False
) -> Optional[str]:
    """
    Convert text to speech using gTTS and convert to proper WAV format.
    """
    try:
        # Generate MP3
        mp3_path = f"{output_filepath}.mp3"
        tts = gTTS(text=input_text, lang=lang, slow=slow)
        tts.save(mp3_path)
        
        # Convert to WAV with proper format
        wav_path = f"{output_filepath}.wav"
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav",
                     parameters=["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"])
        
        # Clean up MP3 file
        os.remove(mp3_path)
        
        # Play audio immediately
        play_audio(wav_path)
        
        return wav_path
        
    except Exception as e:
        logging.error(f"Error in TTS generation: {str(e)}")
        return None

def play_audio(file_path: str) -> None:
    """
    Play audio file using system default player
    """
    try:
        os_name = platform.system()
        if os_name == "Darwin":
            subprocess.run(['afplay', file_path], check=True)
        elif os_name == "Windows":
            subprocess.run(
                ['powershell', '-c', f'(New-Object Media.SoundPlayer "{file_path}").PlaySync();'],
                check=True
            )
        elif os_name == "Linux":
            subprocess.run(['aplay', file_path], check=True)
        else:
            logging.warning(f"Unsupported OS: {os_name}")
            
        logging.info(f"Played audio file: {file_path}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Playback failed: {str(e)}")
    except Exception as e:
        logging.error(f"Playback error: {str(e)}")