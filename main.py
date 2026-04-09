# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt


# # Page configuration

# st.set_page_config(page_title="DR Screening Demo", layout="wide")
# st.title("Diabetic Retinopathy Screening Demo")
# st.write("Upload a retinal fundus image to predict whether DR is present.")

# IMG_SIZE = 224
# MODEL_PATH = "model/best_b0_cbam_newsplit_stage1.keras"



# # Custom CBAM layers

# class ChannelAttention(tf.keras.layers.Layer):
#     def __init__(self, ratio=8, **kwargs):
#         super().__init__(**kwargs)
#         self.ratio = ratio

#     def build(self, input_shape):
#         channels = int(input_shape[-1])
#         self.shared_dense_one = tf.keras.layers.Dense(channels // self.ratio, activation="relu")
#         self.shared_dense_two = tf.keras.layers.Dense(channels)

#     def call(self, inputs):
#         avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
#         max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=False)

#         avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
#         max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

#         out = tf.nn.sigmoid(avg_out + max_out)
#         out = tf.reshape(out, (-1, 1, 1, tf.shape(inputs)[-1]))
#         return inputs * out


# class SpatialAttention(tf.keras.layers.Layer):
#     def __init__(self, kernel_size=7, **kwargs):
#         super().__init__(**kwargs)
#         self.conv = tf.keras.layers.Conv2D(
#             1, kernel_size=kernel_size, strides=1, padding="same", activation="sigmoid"
#         )

#     def call(self, inputs):
#         avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
#         concat = tf.concat([avg_pool, max_pool], axis=-1)
#         attn = self.conv(concat)
#         return inputs * attn


 
# # Load model

# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         custom_objects={
#             "ChannelAttention": ChannelAttention,
#             "SpatialAttention": SpatialAttention
#         }
#     )
#     return model

# model = load_model()



# # Preprocessing

# def crop_black_borders(img, tol=7):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     mask = gray > tol
#     if mask.any():
#         coords = np.argwhere(mask)
#         y0, x0 = coords.min(axis=0)
#         y1, x1 = coords.max(axis=0) + 1
#         img = img[y0:y1, x0:x1]
#     return img


# def preprocess_pil_image(uploaded_image, img_size=224):
#     img = np.array(uploaded_image.convert("RGB"))
#     img = crop_black_borders(img)
#     img = cv2.resize(img, (img_size, img_size))
#     img = img.astype(np.float32)   # no /255
#     return img



# # Grad-CAM helpers

# def find_last_conv_layer(model):
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     raise ValueError("No Conv2D layer found in model.")


# def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
#     if last_conv_layer_name is None:
#         last_conv_layer_name = find_last_conv_layer(model)

#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = tf.maximum(heatmap, 0)
#     max_val = tf.math.reduce_max(heatmap)
#     if max_val > 0:
#         heatmap = heatmap / max_val

#     return heatmap.numpy(), last_conv_layer_name


# def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_uint8 = np.uint8(255 * heatmap)

#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

#     superimposed = np.clip(alpha * heatmap_color + img, 0, 255).astype(np.uint8)
#     return superimposed



# # Upload and predict

# uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     processed_img = preprocess_pil_image(image, IMG_SIZE)
#     input_img = np.expand_dims(processed_img, axis=0)

#     prob = float(model.predict(input_img, verbose=0)[0][0])
#     pred = 1 if prob >= 0.5 else 0

#     class_names = {0: "No DR", 1: "DR"}
#     confidence = prob if pred == 1 else (1 - prob)

#     heatmap, layer_name = make_gradcam_heatmap(input_img, model)
#     gradcam_img = overlay_heatmap_on_image(processed_img.astype(np.uint8), heatmap)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Original Image")
#         st.image(image, use_container_width=True)

#     with col2:
#         st.subheader("Grad-CAM")
#         st.image(gradcam_img, use_container_width=True)

#     st.subheader("Prediction Result")
#     st.write(f"**Predicted Class:** {class_names[pred]}")
#     st.write(f"**Confidence:** {confidence:.4f}")
#     st.write(f"**Raw Probability (DR):** {prob:.4f}")
#     st.write(f"**Grad-CAM Layer Used:** {layer_name}")

#     if pred == 1:
#         st.warning("Screening Result: DR detected")
#     else:
#         st.success("Screening Result: No DR detected")
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="DR Screening System", layout="wide")

st.title("Diabetic Retinopathy Screening System")
st.write(
    "Upload a retinal fundus image to check whether diabetic retinopathy (DR) is likely to be present. "
    "The system also provides a Grad-CAM heatmap to highlight the retinal regions that influenced the prediction."
)

IMG_SIZE = 224
MODEL_PATH = "model/best_b0_cbam_newsplit_stage1.keras"

# -----------------------------
# Custom CBAM layers
# -----------------------------
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced_channels = max(channels // self.ratio, 1)
        self.shared_dense_one = tf.keras.layers.Dense(reduced_channels, activation="relu")
        self.shared_dense_two = tf.keras.layers.Dense(channels)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=False)

        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        out = tf.nn.sigmoid(avg_out + max_out)
        out = tf.reshape(out, (-1, 1, 1, tf.shape(inputs)[-1]))
        return inputs * out


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            1, kernel_size=kernel_size, strides=1, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attn = self.conv(concat)
        return inputs * attn


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention
        },
        compile=False
    )
    return model


model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]
    return img


def preprocess_pil_image(uploaded_image, img_size=224):
    img = np.array(uploaded_image.convert("RGB"))
    img = crop_black_borders(img)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)   # no /255
    return img


# -----------------------------
# Grad-CAM helpers
# -----------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(shape=conv_outputs.shape[:2], dtype=np.float32)

    for i in range(conv_outputs.shape[-1]):
        heatmap += conv_outputs[:, :, i] * pooled_grads[i]

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap, last_conv_layer_name


def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.clip(alpha * heatmap_color + img, 0, 255).astype(np.uint8)
    return superimposed


# -----------------------------
# Risk level and recommendation
# -----------------------------
def get_risk_level(prob):
    if prob < 0.20:
        return "Very Low"
    elif prob < 0.50:
        return "Low"
    elif prob < 0.75:
        return "Moderate"
    elif prob < 0.90:
        return "High"
    else:
        return "Very High"


def get_recommendation(pred, prob):
    if pred == 0:
        if prob < 0.20:
            return "No immediate DR indication. Continue routine eye check-ups."
        else:
            return "Low DR likelihood, but periodic retinal screening is recommended."
    else:
        if prob < 0.75:
            return "Possible DR detected. A follow-up retinal examination is recommended."
        elif prob < 0.90:
            return "DR likely detected. Please consult an eye specialist for further evaluation."
        else:
            return "Strong DR indication. Early ophthalmic consultation is strongly recommended."


# -----------------------------
# Upload and predict
# -----------------------------
uploaded_file = st.file_uploader("Choose a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_img = preprocess_pil_image(image, IMG_SIZE)
    input_img = np.expand_dims(processed_img, axis=0)

    prediction = model.predict(input_img, verbose=0)

    if isinstance(prediction, list):
        prediction = prediction[0]

    prob = float(prediction[0][0])
    pred = 1 if prob >= 0.5 else 0

    class_names = {0: "No Diabetic Retinopathy", 1: "Diabetic Retinopathy Detected"}
    confidence = prob if pred == 1 else (1 - prob)
    risk_level = get_risk_level(prob)
    recommendation = get_recommendation(pred, prob)

    heatmap, layer_name = make_gradcam_heatmap(input_img, model)
    gradcam_img = overlay_heatmap_on_image(processed_img.astype(np.uint8), heatmap)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Retinal Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Model Attention Heatmap (Grad-CAM)")
        st.image(gradcam_img, use_container_width=True)

    st.info(
        "The Grad-CAM heatmap highlights the retinal regions that contributed most to the model's prediction. "
        "Brighter regions indicate stronger influence on the decision."
    )

    st.subheader("Screening Result")
    st.write(f"**Predicted Result:** {class_names[pred]}")
    st.write(f"**Confidence Score:** {confidence:.4f}")
    st.write(f"**Model Probability for DR:** {prob:.4f}")
    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Clinical Recommendation:** {recommendation}")
    st.write(f"**Grad-CAM Layer Used:** {layer_name}")

    if pred == 1:
        st.warning(
            "The uploaded retinal image is predicted as **Diabetic Retinopathy Detected**. "
            "Please note that this is a screening prediction and not a final medical diagnosis."
        )
    else:
        st.success(
            "The uploaded retinal image is predicted as **No Diabetic Retinopathy**. "
            "This result is intended for screening support only."
        )

st.caption(
    "Note: This application is developed for academic project demonstration purposes. "
    "It supports diabetic retinopathy screening and should not be used as a substitute for clinical diagnosis."
)