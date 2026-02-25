import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("Sperm Parts Detection App")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:

    img = Image.open(file)
    img_array = np.array(img)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    result = model(img_array)

    output = img_array.copy()

    head_count = 0
    neck_count = 0
    tail_count = 0

    boxes = result[0].boxes

    if boxes is not None:

        for i in range(len(boxes)):

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            class_id = int(boxes.cls[i])
            name = model.names[class_id].lower()

            if name == "head":
                head_count += 1
                color = (0, 255, 0)

            elif name == "neck":
                neck_count += 1
                color = (255, 0, 0)

            elif name == "tail":
                tail_count += 1
                color = (0, 0, 255)

            else:
                color = (255, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.image(output, use_container_width=True)

    st.write("Heads:", head_count)
    st.write("Necks:", neck_count)
    st.write("Tails:", tail_count)