import streamlit as st
import requests
import os

# Load environment variables
API_KEY_HUGGINGFACE = os.getenv("HUGGINGFACE_API_KEY")
API_KEY_ELEVEN_LABS = os.getenv("ELEVEN_LABS_API_KEY")

def generate_image(user_description):
    url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    headers = {"Authorization": f"Bearer {API_KEY_HUGGINGFACE}"}
    payload = {"inputs": user_description}
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        image_data = response.content
        image_path = "generated_image.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
        return image_path
    else:
        st.error(f"Error generating image: {response.status_code} - {response.text}")
        return None

def understand_image(image_path):
    url = "https://api-inference.huggingface.co/models/your-image-model"
    headers = {"Authorization": f"Bearer {API_KEY_HUGGINGFACE}"}
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    response = requests.post(url, headers=headers, files={"file": image_data})
    
    if response.status_code == 200:
        result = response.json()
        story = result.get("generated_text", "No story generated")
        return story
    else:
        st.error(f"Error understanding image: {response.status_code} - {response.text}")
        return None

def text_to_speech_eleven_labs(input_text):
    url = "https://api.elevenlabs.io/v1/text-to-speech"
    headers = {
        "Authorization": f"Bearer {API_KEY_ELEVEN_LABS}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": input_text,
        "voice": "narrator"  # Specify the voice model
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        audio_path = "output_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    else:
        st.error(f"Error generating audio: {response.status_code} - {response.text}")
        return None

def main():
    st.set_page_config(page_title="Interactive Media Creator", layout="wide")
    st.title("Interactive Media Creator")

    with st.sidebar:
        st.header("Controls")
        image_description = st.text_area("Description for Image Generation", height=100)
        generate_image_btn = st.button("Generate Image")

    col1, col2 = st.columns(2)

    image_path = None
    story = None

    with col1:
        st.header("Comic Art")
        if generate_image_btn and image_description:
            with st.spinner("Generating image..."):
                image_path = generate_image(image_description)
                if image_path:
                    st.image(image_path, caption="Generated Comic Image", use_column_width=True)
                    st.success("Image generated!")
                else:
                    st.error("Failed to generate image.")

    with col2:
        st.header("Story")
        if image_path:
            with st.spinner("Understanding image and generating story..."):
                try:
                    story = understand_image(image_path)
                    if story:
                        st.write(story)
                        audio_path = text_to_speech_eleven_labs(story)
                        if audio_path:
                            st.audio(audio_path, format="audio/wav")
                            st.success("Audio generated from story!")
                        else:
                            st.error("Failed to generate audio.")
                    else:
                        st.error("Failed to generate story.")
                except Exception as e:
                    st.error(f"Failed to process image: {e}")

if __name__ == "__main__":
    main()
