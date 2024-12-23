# Install necessary libraries in requirements.txt
# Import libraries
import whisper
import os
from gtts import gTTS
import gradio as gr
from groq import Groq

# Load Whisper model for transcription
model = whisper.load_model("base")

# Set up Groq API client (ensure GROQ_API_KEY is set in your environment)
GROQ_API_KEY='YOR_API_KEY_HERE'
client = Groq(api_key=GROQ_API_KEY)


# Function to get the LLM response from Groq

# Function to get the LLM response from Groq
def get_llm_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",  # Replace with your desired model
    )
    return chat_completion.choices[0].message.content

# Function to convert text to speech using gTTS
def text_to_speech(text, output_audio="output_audio.mp3"):
    tts = gTTS(text)
    tts.save(output_audio)
    return output_audio

# Main chatbot function to handle audio input and output
def chatbot(audio):
    # Step 1: Transcribe the audio using Whisper
    result = model.transcribe(audio)
    user_text = result["text"]

    # Step 2: Get LLM response from Groq
    response_text = get_llm_response(user_text)

    # Step 3: Convert the response text to speech
    output_audio = text_to_speech(response_text)
    
    return response_text, output_audio

# Customize the appearance of the interface (removed layout and compact theme)
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(
        type="filepath",  # This allows file upload or microphone input
        label="Speak Now", 
        interactive=True
    ),  # Input from mic or file
    outputs=[
        gr.Textbox(
            label="Response Text",
            placeholder="Your response will appear here...",
            lines=4
        ),  # Output: response text
        gr.Audio(
            type="filepath", 
            label="Audio Response"
        ),  # Output: response audio
    ],
    live=True,
    title="Audio Chatbot",  # Title of the app
    description="This is an AI-powered chatbot that responds based on your speech input. Speak into the microphone and get an audio response!",
)

# Launch the Gradio app
iface.launch()
