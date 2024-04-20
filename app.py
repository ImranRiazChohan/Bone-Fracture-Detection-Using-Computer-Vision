
# Python In-built packages
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# External packages
import streamlit as st

# Local Modules
import setting
import helper

# Setting page layout
st.set_page_config(
    page_title="Bone Fracture",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# st.title("Bone Fracture Detection")
# Center-align text using Markdown and HTML
st.markdown("<h1 style='text-align: center;'>Bone Fracture Detection</h1>", unsafe_allow_html=True)
model_path = Path(setting.Detection_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Upload Image
st.markdown("<h4 style='text-align: left;'>Upload Image</h4>", unsafe_allow_html=True)
image_path=st.file_uploader('Choose an image...',type=['jpg','jpeg','png'])    

if image_path is not None:           
    
    uploaded_image = Image.open(image_path)
    st.image(image_path, caption="Uploaded Image",use_column_width=True)
        
    if st.button('Apply Detection Model'):
        results = list(model(uploaded_image))
        result=results[0]
        print(result)
        res_plotted = result.plot(line_width=1,labels=True)[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image',use_column_width=True)
