import streamlit as st
from PIL import Image
import requests
import os
from transformers import  AutoFeatureExtractor, AutoModelForImageClassification
from diffusers import StableDiffusionPipeline
import torch

# Environment variables
API_KEY_ELEVEN_LABS = os.getenv("ELEVEN_LABS_API_KEY")
STABLE_DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
VIT_MODEL = "google/vit-base-patch16-224-in21k"


def generate_image(user_description):
    model = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    image = model(user_description).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path


def understand_image(image_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(VIT_MODEL)
    model = AutoModelForImageClassification.from_pretrained(VIT_MODEL)

    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    return f"Predicted class ID: {predicted_class_id}"  # Adjust as needed for more detailed descriptions


def text_to_speech_eleven_labs(input_text, output_path):
    headers = {
        "Authorization": f"Bearer {API_KEY_ELEVEN_LABS}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": input_text,
        "voice": "default"  # Specify the voice model if applicable
    }

    response = requests.post("https://api.elevenlabs.io/v1/text-to-speech", json=payload, headers=headers)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def main():
    st.set_page_config(page_title="Interactive Media Creator", layout="wide")
    st.title("Interactive Media Creator")

    with st.sidebar:
        st.header("Controls")
        image_description = st.text_area("Description for Image Generation", height=100)
        generate_image_btn = st.button("Generate Image")

    col1, col2 = st.columns(2)

    image_path = None

    with col1:
        st.header("Comic Art")
        if generate_image_btn and image_description:
            with st.spinner("Generating image..."):
                try:
                    image_path = generate_image(image_description)
                    if image_path:
                        st.image(image_path, caption="Generated Comic Image", use_column_width=True)
                        st.success("Image generated!")
                    else:
                        st.error("Failed to generate image.")
                except Exception as e:
                    st.error(f"Failed to generate image: {e}")

    with col2:
        st.header("Story")
        if image_path:
            with st.spinner("Creating a story..."):
                try:
                    understood_text = understand_image(image_path)
                    if understood_text:
                        audio_path = text_to_speech_eleven_labs(understood_text, "output_audio.wav")
                        if audio_path:
                            st.audio(audio_path, format="audio/wav")
                            st.success("Audio generated from image understanding!")
                        else:
                            st.error("Failed to generate audio.")
                    else:
                        st.error("Failed to generate story.")
                except Exception as e:
                    st.error(f"Failed to process image: {e}")


if __name__ == "__main__":
    main()
