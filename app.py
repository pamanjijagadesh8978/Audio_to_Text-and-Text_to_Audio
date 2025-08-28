# filename: app.py
import streamlit as st
import requests
import io
import json

# Set a single page configuration for the entire app
st.set_page_config(
    page_title="Text to Speech & Speech to Text", 
    layout="wide"
)

st.title("🗣️ AI Audio Services")
st.markdown("Use this app to generate speech from text or transcribe audio to text.")

# Define the FastAPI backend URL once to avoid repetition
BACKEND_URL = "http://127.0.0.1:8000"

# Use tabs to separate the two main functionalities
tts_tab, stt_tab = st.tabs(["Text to Speech", "Speech to Text"])

# --- Text to Speech (TTS) Tab ---
with tts_tab:
    st.header("Text to Speech")
    st.markdown("Enter text and generate speech using the Kokoro TTS model.")
    
    # Input text box for TTS
    input_text = st.text_area("Enter your text here:", "This is a simple test for the Kokoro TTS model.")

    # Button to trigger the TTS generation
    if st.button("Generate Audio"):
        if input_text:
            with st.spinner('Generating audio...'):
                try:
                    # The payload for the POST request
                    payload = {
                        "text": input_text,
                        "voice": "af_heart"
                    }
                    
                    # Make the POST request to the TTS API endpoint
                    tts_api_url = f"{BACKEND_URL}/tts"
                    response = requests.post(tts_api_url, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        st.audio(response.content, format="audio/wav")
                        st.success("Audio generated successfully!")
                    else:
                        st.error(f"Error from API: {response.text}")

                except requests.exceptions.Timeout:
                    st.error("The request timed out. The server might be busy. Please try again with a shorter text.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the API. Please ensure the server is running. Error: {e}")
        else:
            st.warning("Please enter some text to generate audio.")

# --- Speech to Text (STT) Tab ---
with stt_tab:
    st.header("Speech to Text")
    st.markdown("Record your voice, and the FastAPI backend will transcribe it.")
    
    # Use the streamlit-audiorecorder widget
    from audio_recorder_streamlit import audio_recorder
    st_audio_bytes = audio_recorder()

    if st_audio_bytes:
        st.audio(st_audio_bytes, format="audio/wav")

        st.markdown("### Transcribing...")
        
        try:
            # Send the audio data to the STT backend
            stt_api_url = f"{BACKEND_URL}/transcribe/"
            files = {"file": ("audio.wav", io.BytesIO(st_audio_bytes), "audio/wav")}
            response = requests.post(stt_api_url, files=files)

            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get("transcribed_text", "No text found.")
                st.success("✅ **Transcription Complete:**")
                st.write(f'**You said:** "{transcribed_text}"')
            else:
                error_detail = response.json().get("detail", "Unknown error.")
                st.error(f"❌ **Error from Backend:** {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("❌ **Connection Error:** Could not connect to the backend server. Make sure it's running.")
        except json.JSONDecodeError:
            st.error("❌ **JSON Error:** Received an invalid response from the backend.")
        except Exception as e:
            st.error(f"❌ **An unexpected error occurred:** {e}")