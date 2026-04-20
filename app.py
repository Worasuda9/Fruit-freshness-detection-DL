import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Fresh", "Rotten"]

# LOAD MODEL
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("fruit_model3_finetuned.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# PREDICTION FUNCTION
def predict(image):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    label = CLASS_NAMES[pred.item()]
    conf = confidence.item() * 100
    return label, conf

def show_result(label, conf):
    if label == "Fresh":
        st.success(f"Prediction: {label} ({conf:.2f}%)")
    else:
        st.error(f"Prediction: {label} ({conf:.2f}%)")

# UI
st.title("Fruit Freshness Detection")

mode = st.radio("Select input mode:", ["Upload Image", "Snapshot"], horizontal=True)

# UPLOAD MODE
if mode == "Upload Image":
    st.write("Upload an image to check if the fruit is Fresh or Rotten")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        label, conf = predict(image)
        show_result(label, conf)

# SNAPSHOT MODE
elif mode == "Snapshot":
    st.write("Take a photo using your camera")
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        label, conf = predict(image)
        show_result(label, conf)
