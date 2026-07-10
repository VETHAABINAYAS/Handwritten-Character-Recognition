import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cnn_model.keras")

st.set_page_config(page_title="Handwritten Digit Recognition")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9).")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    # Threshold
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find largest contour
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        digit = thresh[y:y+h, x:x+w]

        digit = cv2.resize(digit, (20,20))

        canvas = np.zeros((28,28), dtype=np.uint8)

        canvas[4:24,4:24] = digit

        canvas = canvas.astype("float32") / 255.0

        st.subheader("Processed Image")
        st.image(canvas, use_container_width=True)

        input_img = canvas.reshape(1,28,28,1)

        prediction = model.predict(input_img, verbose=0)

        digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        probs = prediction[0]

        for i in range(10):
            st.progress(float(probs[i]))
            st.write(f"{i} : {probs[i]*100:.2f}%")
