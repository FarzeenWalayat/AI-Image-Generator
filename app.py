#import streamlit as st

# Title of your app
#st.title('Image Generation App')

# Add a text input for the user to enter a prompt
#prompt = st.text_input("Enter a prompt for the image:")

# Display the entered prompt
#if prompt:
#   st.write(f"Your entered prompt: {prompt}")
#---------------------------------------------------------------------------------
#import streamlit as st
#import requests
#from PIL import Image
#from io import BytesIO

# Title of the app
#st.title('Image Generation App')

# Add a text input for the user to enter a prompt
#prompt = st.text_input("Enter a prompt for the image:")

# Add a button to generate the image
#if st.button("Generate Image"):
 #   if prompt:
        # Hugging Face API URL for Stable Diffusion model
        #api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2.1"
        #api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
        #api_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4-original"
        #api_url = "https://huggingface.co/stabilityai/stable-diffusion-3.5-large"

  #      api_url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers"
       # https: // huggingface.co / stabilityai / stable - diffusion - 3 - medium - diffusers

  #    headers = {
    #        "Authorization": f"Bearer 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"
       # }
   #     payload = {
    #        "inputs": prompt,
     #   }

        # Make the API request
     #   response = requests.post(api_url, headers=headers, json=payload)

        # Check if the request was successful
   #     if response.status_code == 200:
            # Get the image from the response
    #        image = Image.open(BytesIO(response.content))
     #       st.image(image, caption="Generated Image", use_column_width=True)
      #  else:
       #     st.error(f"Error: {response.status_code}. Could not generate image.")
    #else:
     #   st.warning("Please enter a prompt!")

#-------------------------------------------------------------------------------------------

import os
import sqlite3
from huggingface_hub import InferenceClient
from PIL import Image
import streamlit as st
from datetime import datetime
import torch
from transformers import CLIPProcessor, CLIPModel
import base64
import sqlite3
############TO CHECK DATA IN DATABASE#####################################
#conn = sqlite3.connect("prompts_history.db")
#c = conn.cursor()
#c.execute("SELECT * FROM prompt_history")
#records = c.fetchall()
#conn.close()

#st.write("Database records found:", len(records))
#for record in records:
#    st.write(record)

###########################










# ------------------- Setup ---------------------
DB_PATH = "prompts_history.db"
IMAGES_DIR = "generated_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Hugging Face client (replace with your own token if needed)
client = InferenceClient(
    model="stabilityai/stable-diffusion-3-medium-diffusers",
    token="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"
)
##########################################################

# Set background image from local project folder manually
selected_bg_path = "backgrounds/Blue and White Simple Nature Flower Water Quotes Desktop Wallpaper.png"

# = "backgrounds/Blue Simple Keep Calm Desktop Wallpaper.png"
#selected_bg_path = "backgrounds/Blue Gradient Modern Professional Company Zoom Virtual Background.png"


def get_base64_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

bg_base64 = get_base64_background(selected_bg_path)
#######STYYLE##############
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{bg_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}
    .app-title {{
        text-align: center;
        font-size: 250px;
        font-weight: bold;
        color: #3f51b5;  /* Deep Indigo */
        margin-bottom: 20px;
    }}

    .stTextInput > div > input {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 12px;
        font-size: 16px;
        border: 1px solid #b2dfdb;
        border-radius: 5px;
    }}

    .stButton > button {{
        background-color: #00695c !important;
        color: white !important;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
        transition: 0.3s ease;
    }}

    .stButton > button:hover {{
        background-color: #004d40 !important;  /* Correct teal hover */
        color: white !important;
    }}

    .stMarkdown {{
        background: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    </style>
""", unsafe_allow_html=True)

#########################################################
# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --------------- Database Utils -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            image_url TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            evaluation TEXT,
            clip_score REAL DEFAULT 0.0
        )
    ''')
    conn.commit()
    conn.close()

def add_prompt(prompt, image_path, clip_score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO prompt_history (prompt, image_url, clip_score) VALUES (?, ?, ?)",
              (prompt, image_path, clip_score))
    conn.commit()
    conn.close()

# ------------------ Scoring --------------------
def compute_clip_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = torch.nn.functional.cosine_similarity(
            outputs.image_embeds, outputs.text_embeds
        )
    return float(similarity.item())

# ------------------ UI --------------------------
init_db()
#st.markdown("<h1 style='text-align:center;color:green;'>AI Image Generator</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>AI Image Generator</h1>", unsafe_allow_html=True)


prompt = st.text_input("Enter a prompt:")

if st.button("Generate Image"):
    if prompt:
        try:
            image = client.text_to_image(prompt)
            filename = f"{prompt.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            image_path = os.path.join(IMAGES_DIR, filename)
            image.save(image_path)
            score = compute_clip_score(image_path, prompt)
            add_prompt(prompt, image_path, score)
            st.image(image_path, caption="Generated Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Prompt cannot be empty.")

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if st.button("Show/Hide Prompt History"):
    st.session_state.show_history = not st.session_state.show_history

# ------------------ Prompt History --------------------------
if st.session_state.show_history:
    st.subheader("Prompt History 🔁")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM prompt_history ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()

    for record in history:
        if len(record) == 7:
            rid, prompt_text, path, ts, _, evaluation, score = record

            st.markdown(f"**🕒 Time:** {ts}  \n📌 **Prompt:** `{prompt_text}`  \n🧠 **AI Match Score:** `{score:.2f}`")

            if os.path.exists(path):
                st.image(path, caption="Generated Image", use_container_width=True)

                eval_key = f"eval_{rid}"
                eval_input = st.text_area("Evaluation:", value=evaluation or "", key=eval_key)

                if st.button("Save Evaluation", key=f"save_{rid}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE prompt_history SET evaluation = ? WHERE id = ?", (eval_input, rid))
                    conn.commit()
                    conn.close()
                    st.success("Evaluation saved!")

                    st.rerun()

                if st.button("Delete", key=f"delete_{rid}"):
                    if os.path.exists(path):
                        os.remove(path)
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("DELETE FROM prompt_history WHERE id = ?", (rid,))
                    conn.commit()
                    conn.close()
                    st.rerun()
        else:
            st.warning(f"⚠️ Skipping invalid record: {record}")
    # ---- Clear History Section ----
    st.markdown("---")
    st.subheader("🧹 Clear All Prompt History")

    if st.button("Delete All History"):
        conn = sqlite3.connect("prompts_history.db")
        c = conn.cursor()
        c.execute("DELETE FROM prompt_history")
        conn.commit()
        conn.close()

        # Delete all images too
        for file in os.listdir("generated_images"):
            os.remove(os.path.join("generated_images", file))

        st.success("All prompt history and images have been deleted.")
        st.rerun()  # Refresh to update view
