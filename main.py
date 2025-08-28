# filename: main.py
import io
import os
import tempfile
import soundfile as sf
import whisper
import numpy as np
from fastapi import FastAPI, Response, File, UploadFile, HTTPException
from pydantic import BaseModel
from kokoro import KPipeline

# Initialize the FastAPI app
app = FastAPI()

# Pydantic models to define the request body structure
class TextRequest(BaseModel):
    """
    Request model for the TTS API.
    """
    text: str
    voice: str = 'af_heart'

# --- TTS model initialization ---
try:
    tts_pipeline = KPipeline(lang_code='a')
except Exception as e:
    print(f"Error initializing KPipeline: {e}")
    tts_pipeline = None

# --- ASR (Whisper) model initialization ---
# "small" is a good balance of speed and accuracy
asr_model = whisper.load_model("small")

@app.post("/tts", summary="Generate speech from text", description="Generates audio from a given text string using the Kokoro TTS model.")
async def text_to_speech(request: TextRequest):
    """
    Endpoint to generate speech from a given text string.

    Args:
        request (TextRequest): The request body containing the text and voice.

    Returns:
        Response: An audio file in WAV format.
    """
    if tts_pipeline is None:
        return Response(content="Model not available. Check server logs.", status_code=500)

    try:
        generator = tts_pipeline(request.text, voice=request.voice)
        all_audio_chunks = [audio_chunk for _, _, audio_chunk in generator]

        if not all_audio_chunks:
            return Response(content="No audio generated for the provided text.", status_code=400)

        audio_data = np.concatenate(all_audio_chunks)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format='WAV')
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return Response(content=f"Internal Server Error: {e}", status_code=500)

@app.post("/transcribe/", summary="Transcribe audio to text", description="Transcribes an audio file to text using OpenAI Whisper.")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file to text using OpenAI Whisper.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        result = asr_model.transcribe(tmp_file_path)

        return {"transcribed_text": result["text"].strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        # Clean up the temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)