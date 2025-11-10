# main page
# imports
import streamlit as st
from PIL import Image
import base64
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import tempfile
import random
import base64
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from PIL import Image
from moviepy.editor import (
    AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips
)
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
from moviepy.video.fx.all import resize
from moviepy.config import change_settings
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# templates for making stories
TEMPLATES = [
    "Deep in Uganda's lush forests, the {name} flashes its {color_phrase} feathers. {desc} It dances on branches at dawn, a true jewel of the Pearl of Africa.",
    "Along the Nile's banks, the {name} stands tall with {color_phrase} plumage. {desc} Fishermen smile when they hear its melodic call at sunrise.",
    "In Queen Elizabeth National Park, the {name} soars above acacia trees. {desc} Its {color_phrase} wings catch the golden light of the savanna.",
    "Near Lake Victoria, the {name} perches quietly. {desc} Children in fishing villages know its {color_phrase} colors mean good luck for the day.",
    "High in the Rwenzori Mountains, the {name} sings through mist. {desc} Its {color_phrase} feathers shine like emeralds in the cloud forest.",
    "In Murchison Falls, the {name} glides over roaring waters. {desc} Tourists gasp at its {color_phrase} beauty against the dramatic backdrop.",
    "Among papyrus swamps, the {name} wades gracefully. {desc} Its long legs and {color_phrase} crest make it the king of the wetlands.",
    "At sunset in Kidepo Valley, the {name} calls across the plains. {desc} Its {color_phrase} silhouette is a symbol of Uganda's wild heart.",
    "In Bwindi's ancient rainforest, the {name} flits between vines. {desc} Gorilla trackers pause to admire its {color_phrase} brilliance.",
    "By the shores of Lake Mburo, the {name} reflects in calm waters. {desc} Its {color_phrase} feathers mirror the peace of the savanna night."
]

# loading bird data
@st.cache_resource
def load_bird_data():
    pth_path = "bird_data.pth"
    if not Path(pth_path).exists():
        st.error(f"Missing `{pth_path}`. Upload it to your app folder.")
        st.stop()
    return torch.load(pth_path, map_location="cpu")

bird_db = load_bird_data()

# building a story
def generate_story(name, desc, colors):
    import random
    color_phrase = ", ".join(colors) if colors else "vibrant"
    desc = desc.strip().capitalize() if desc else "A fascinating bird with unique habits."
    tmpl = random.choice(TEMPLATES) #random template
    return tmpl.format(name=name, color_phrase=color_phrase, desc=desc)
# Text To Speech
def natural_tts(text, path):
    try:
        from gtts import gTTS
        tts = gTTS(text, lang='en')
        tts.save(path)
        return path
    except Exception as e:
        raise RuntimeError(f"TTS failed: {e}")
# Ken Burns
def ken_burns_clip(img_path, duration=4.0):
    clip = ImageClip(img_path).set_duration(duration)
    w, h = clip.size
    zoom = 1.15
    clip = clip.resize(lambda t: 1 + (zoom - 1) * (t / duration))
    clip = clip.set_position(lambda t: (
        "center" if t < duration * 0.6 else (w * 0.05 * (t - duration * 0.6) / (duration * 0.4)),
        "center"
    ))
    return clip.fadein(0.3).fadeout(0.3)
# Video properties
def create_video(image_paths, audio_path, output_path):
    audio = AudioFileClip(audio_path)
    audio = audio_fadein(audio, 0.6).audio_fadeout(1.2)

    duration_per_img = 4.0
    total = duration_per_img * len(image_paths)

    if audio.duration < total:
        loops = int(total / audio.duration) + 1
        audio = concatenate_audioclips([audio] * loops).subclip(0, total)
    else:
        audio = audio.subclip(0, total)

    clips = [ken_burns_clip(img, duration_per_img) for img in image_paths]
    video = concatenate_videoclips(clips, method="compose").set_audio(audio)
    video = video.resize(height=720)
    video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="medium")
    return output_path
def _set_background_glass(img_path: str = "ugb1.png"):
    """background ub1.png photo
    """
    try:
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp .main .block-container {{
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 1rem 1.5rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

# background ub1.png
_set_background_glass("ugb1.png")

# Model loading and prediction
@st.cache_resource
def load_model():
    """Load the ResNet34 model with trained weights"""
    try:
        model = models.resnet34(weights=None)
        num_classes = 220  # 220 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # weights
        if os.path.exists("resnet34_weights.pth"):
            model.load_state_dict(torch.load("resnet34_weights.pth", map_location=torch.device('cpu')))
            model.eval()
            return model
        else:
            st.error("Model file not found: resnet34_weights.pth")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_label_map():
    """Load the label mapping from JSON file"""
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json", "r") as f:
                label_map = json.load(f)
            idx_to_label = {v: k for k, v in label_map.items()}
            return idx_to_label
        else:
            st.error("Label map file not found: label_map.json")
            return None
    except Exception as e:
        st.error(f"Error loading label map: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_species(model, label_map, image):
    """Predict bird species from image - returns top prediction only"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor) 
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top prediction (dim=1 is the class dimension)
        top_prob, top_index = torch.max(probabilities, dim=1)
        
        idx = top_index.item()
        prob = top_prob.item()
        bird_name = label_map.get(idx, f"Class {idx}")
        
        result = {
            'species': bird_name,
            'confidence': prob * 100
        }
        
        return result
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Load model and label map
model = load_model()
label_map = load_label_map()

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""<style>
:root {
  --bg: #ffffff;
  --card: #ffffff;
  --muted: #1f2937; /* slate-800 for strong contrast on light bg */
  --text: #0f172a;  /* slate-900 as primary text */
  --brand: #16a34a;
  --brand-2: #0e7490;
  --brand-3: #6d28d9;
  --ring: rgba(15,23,42,0.25);
}
html, body, .stApp { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji"; color: var(--text); }
.stApp::before, .stApp::after {
  content: "";
  position: fixed;
  inset: auto auto 10% -10%;
  width: 40vw;
  height: 40vw;
  background: radial-gradient(closest-side, rgba(34,197,94,0.18), transparent 65%);
  filter: blur(40px);
  z-index: 0;
  pointer-events: none;
}
.stApp::after {
  inset: -15% -10% auto auto;
  width: 35vw;
  height: 35vw;
  background: radial-gradient(closest-side, rgba(6,182,212,0.16), transparent 65%);
}
.stApp .main .block-container { position: relative; z-index: 1; }
.hero {
  background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(15,23,42,0.55));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px;
  padding: 2.25rem;
  margin: 0.75rem 0 1.5rem 0;
  box-shadow: 0 12px 30px rgba(2,6,23,0.45);
}
.hero-title {
  font-family: Poppins, Inter, system-ui;
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: clamp(1.8rem, 2.5vw + 1.2rem, 3.25rem);
  margin: 0 0 .35rem 0;
  background: linear-gradient(90deg, #0f172a, #1f2937 45%, #0b1220 85%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.6;
}
.hero-badges { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .85rem; }
.badge {
  font-size: .8rem;
  color: #0f172a;
  background: rgba(15,23,42,0.08);
  padding: .35rem .6rem;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,0.15);
  font-weight: 600;
}
.card {
  background: #ffffff;
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 16px;
  padding: 1.25rem;
  box-shadow: 0 10px 24px rgba(2,6,23,0.06);
  color: var(--text);
}
.card h4 { color: var(--text); margin: 0 0 .5rem 0; font-weight: 700; }
.card .hint { color: var(--muted); font-size: .92rem; margin-bottom: .75rem; }
.stButton > button {
  background: linear-gradient(135deg, var(--brand), #16a34a);
  color: white;
  border: 0;
  padding: .7rem 1rem;
  border-radius: 10px;
  width: 100%;
  font-weight: 600;
  box-shadow: 0 6px 14px rgba(16,185,129,0.28);
  transition: transform .08s ease, filter .2s ease, box-shadow .2s ease;
}
.stButton > button:hover { filter: brightness(1.05); box-shadow: 0 10px 18px rgba(16,185,129,0.32); }
.stButton > button:active { transform: translateY(1px); }
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(2,6,23,0.15);
  background: #f8fafc;
  transition: border-color .2s ease, background .2s ease, box-shadow .2s ease;
  border-radius: 14px;
}
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(15,23,42,0.45);
  box-shadow: 0 8px 20px rgba(2,6,23,0.08);
  background: #f1f5f9;
}
[data-testid="stFileUploader"] section > div { color: #0f172a !important; }
[data-testid="stFileUploader"] label { color: #0f172a !important; font-weight: 600; }
[data-testid="stCameraInputLabel"] { color: #0f172a !important; font-weight: 600; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.fade { animation: fadeUp .4s ease both; }
.input-section {
  background: rgba(2,6,23,0.75);
  border-radius: 14px;
  padding: 1rem;
  margin: 0.5rem 0;
  border: 1px solid rgba(255,255,255,0.06);
}
.section-title {
  color: #0f172a;
  font-size: 1.05rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}
.result-card {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  border: 2px solid rgba(16,185,129,0.2);
  border-radius: 12px;
  padding: 1.25rem;
  margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(16,185,129,0.1);
}
.result-title {
  color: #0f172a;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}
.result-item {
  background: #ffffff;
  border-left: 4px solid #16a34a;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.result-species {
  color: #0f172a;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.result-confidence {
  color: #16a34a;
  font-size: 0.95rem;
  font-weight: 500;
}
</style>""", unsafe_allow_html=True)
try:
    _logo = Image.open("ugb1.png")
    _w, _h = _logo.size
    _new_w = max(1, _w // 2) 
    _new_h = max(1, _h // 2)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)
    with st.container():
        logo_col, text_col = st.columns([1, 3])
        with logo_col:
            st.image(_logo_small, use_column_width=False)
        with text_col:
            st.markdown("<div class='hero-title'>Bird Spotter</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-sub'>Identify birds from photos in seconds. Upload an image or use your camera to discover species, with a beautiful, distraction-free interface.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='hero-badges'><span class='badge'>Identify</span><span class='badge'>Visualize</span><span class='badge'>Learn</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
except Exception:
    pass

# Main content container
with st.container():
    st.markdown(
        """
        <div style='text-align:center; margin: .5rem 0 1rem;'>
            <p style='color:#0f172a; margin:0; font-weight:700; font-size:1rem; letter-spacing:.01em;'>
                Choose how you want to identify a bird
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

    with tab_upload:
        st.markdown("<h4>📁 Upload Image</h4>", unsafe_allow_html=True)
        st.markdown("<div class='hint'>PNG or JPEG. Clear, close-up shots improve results.</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a bird image", type=['png', 'jpg', 'jpeg'], key="uploader_file")
        if uploaded_file is not None:
            if 'upload_result' in st.session_state:
                del st.session_state.upload_result
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width='stretch')
            
            if st.button("Identify Specie", key="identify_specie_upload_button"):
                if model is not None and label_map is not None:
                    with st.spinner("🔍 Analyzing image..."):
                        result = predict_species(model, label_map, image)
                    
                    if result:
                        st.session_state.upload_result = result
                        st.session_state.upload_image = image
                    else:
                        st.error("Failed to predict species. Please try again.")
                else:
                    st.error("Model or label map not loaded. Please check if the files exist.")

            
            if 'upload_result' in st.session_state and st.session_state.upload_result:
                result = st.session_state.upload_result
                st.markdown("<div class='result-title'>🦅 Identification Result</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='result-item'>
                    <div class='result-species'>{result['species']}</div>
                    <div class='result-confidence'>Confidence: {result['confidence']:.3f}%</div>
                </div>
                """, unsafe_allow_html=True)

            bird_name = st.text_input("Know more about the predicted specie. Write the specie name to Generate video", placeholder="e.g. African Jacana").strip().title()
            

            if bird_name:
                if bird_name not in bird_db:
                    st.error(f"**{bird_name}** not found.")
                else:
                    if st.button("Generate Video", type="primary"):
                        with st.spinner("Generating..."):
                            data = bird_db[bird_name]
                            story = generate_story(bird_name, data["desc"], data["colors"])
                            tmp = tempfile.mkdtemp()
                            img_paths = []
                            for i, b64 in enumerate(data["images_b64"]):
                                img_data = base64.b64decode(b64)
                                img = Image.open(BytesIO(img_data))
                                p = os.path.join(tmp, f"img_{i}.jpg")
                                img.save(p, "JPEG")
                                img_paths.append(p)
                            audio_path = os.path.join(tmp, "voice.mp3")
                            natural_tts(story, audio_path)
                            out_path = os.path.join(tmp, f"{bird_name.replace(' ', '_')}.mp4")
                            create_video(img_paths, audio_path, out_path)
                            st.video(out_path)
                            with open(out_path, "rb") as f:
                                st.download_button("Download Video", f, f"{bird_name}.mp4", "video/mp4")

                            shutil.rmtree(tmp, ignore_errors=True)
                            st.success("Done!")
            with st.expander("Available birds"):
                st.write(", ".join(sorted(bird_db.keys())))                


            

    with tab_camera:
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 1

        st.markdown("<h4>Take Picture</h4>", unsafe_allow_html=True)

    if not st.session_state.camera_active:
        try:
            _placeholder_path = "ub2.png"
            if os.path.exists(_placeholder_path):
                with open(_placeholder_path, "rb") as _f:
                    _data = _f.read()
                _b64 = base64.b64encode(_data).decode()
                _img_html = (
                    f"<img src=\"data:image/png;base64,{_b64}\" "
                    "style=\"width:100%; aspect-ratio:4/3; min-height:280px; object-fit:cover; "
                    "border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);\"/>")
                st.markdown(_img_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:linear-gradient(180deg,#0b1220,#0b1220 60%, #0f172a); border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:linear-gradient(180deg,#0b1220,#0b1220 60%, #0f172a); border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                unsafe_allow_html=True,
            )

        def _start_camera():
            st.session_state.camera_active = True
            st.session_state.zoom_level = 1

        st.button("Start Camera", key="use_camera_button", on_click=_start_camera)

    if st.session_state.camera_active:
        camera_photo = st.camera_input("Take a photo", key="camera_input")

        if camera_photo is not None:
            if 'camera_result' in st.session_state:
                del st.session_state.camera_result

            image = Image.open(camera_photo)
            st.session_state.camera_image = image

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Captured Photo**")
            with col2:
                zoom_options = [1, 2, 4, 8, 10]
                zoom_labels = [f"x{int(z)}" if z > 1 else "Original" for z in zoom_options]
                selected_idx = st.selectbox(
                    "Zoom",
                    options=range(len(zoom_options)),
                    format_func=lambda i: zoom_labels[i],
                    index=zoom_options.index(st.session_state.zoom_level),
                    key="zoom_selector"
                )
                st.session_state.zoom_level = zoom_options[selected_idx]

            display_image = image.copy()
            if st.session_state.zoom_level > 1:
                scale = st.session_state.zoom_level
                new_size = (int(image.width * scale), int(image.height * scale))
                display_image = image.resize(new_size, Image.Resampling.LANCZOS)

            st.image(display_image, use_column_width=True)

            if st.button("Identify Specie", key="identify_specie_camera_button"):
                if model is not None and label_map is not None:
                    with st.spinner("Analyzing image..."):
                        result = predict_species(model, label_map, image)
                    if result:
                        st.session_state.camera_result = result
                    else:
                        st.error("Failed to predict species. Please try again.")
                else:
                    st.error("Model or label map not loaded.")

            if 'camera_result' in st.session_state and st.session_state.camera_result:
                result = st.session_state.camera_result
                st.markdown("<div class='result-title'>Identification Result</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='result-item'>
                    <div class='result-species'>{result['species']}</div>
                    <div class='result-confidence'>Confidence: {result['confidence']:.3f}%</div>
                </div>
                """, unsafe_allow_html=True)

            bird_name = st.text_input(
                "Know more about the predicted specie. Write the specie name to Generate video",
                placeholder="e.g. African Jacana"
            ).strip().title()

            if bird_name:
                if bird_name not in bird_db:
                    st.error(f"**{bird_name}** not found.")
                else:
                    if st.button("Generate Video", type="primary"):
                        with st.spinner("Generating..."):
                            data = bird_db[bird_name]
                            story = generate_story(bird_name, data["desc"], data["colors"])
                            tmp = tempfile.mkdtemp()
                            img_paths = []
                            for i, b64 in enumerate(data["images_b64"]):
                                img_data = base64.b64decode(b64)
                                img = Image.open(BytesIO(img_data))
                                p = os.path.join(tmp, f"img_{i}.jpg")
                                img.save(p, "JPEG")
                                img_paths.append(p)
                            audio_path = os.path.join(tmp, "voice.mp3")
                            natural_tts(story, audio_path)
                            out_path = os.path.join(tmp, f"{bird_name.replace(' ', '_')}.mp4")
                            create_video(img_paths, audio_path, out_path)
                            st.video(out_path)
                            with open(out_path, "rb") as f:
                                st.download_button("Download Video", f, f"{bird_name}.mp4", "video/mp4")
                            shutil.rmtree(tmp, ignore_errors=True)
                            st.success("Done!")

            with st.expander("Available birds"):
                st.write(", ".join(sorted(bird_db.keys())))

        if st.button("Stop Camera", key="stop_camera_button"):
            st.session_state.camera_active = False
            for key in ['camera_image', 'camera_result', 'zoom_level']:
                if key in st.session_state:
                    del st.session_state[key]

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; color:#334155; margin-top: 1rem; font-size:.9rem;'>
            Built for the Love of Nature
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
        <style>
        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, #0f172a, #1f2937);
        color: white;
        text-align: center;
        padding: 1rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        z-index: 999;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.15);
        border-top: 1px solid rgba(255,255,255,0.1);}
        .footer a {
        color: #86efac;
        text-decoration: none;
        font-weight: 600;}
        .footer a:hover {
        text-decoration: underline;}
        </style>

        <div class="footer">
        <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <span style="color:#f87171;">Developed by</span><div><i>&bull;Kasozi David</i><i>&nbsp;&nbsp;&bull;Namuzibwa Laurinda</i><i>&nbsp;&nbsp;&bull;Zziwa Mark Alex</i></div>
        </div>
        </div>
        """, unsafe_allow_html=True)
