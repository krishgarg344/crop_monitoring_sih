# cropvision_dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- PAGE CONFIG ---
st.set_page_config(page_title="CropVision 🌾", layout="wide")

# --- HEADER ---
st.markdown(
    """
    <style>
    .big-title { font-size:40px !important; font-weight:700; color:#2E7D32; }
    .sub-header { font-size:20px !important; color:#555; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown("<p class='big-title'>🌾 CropVision Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered insights for crop health, soil, and pest risk monitoring</p>", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model("model/crop_health_model.keras")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_keras_model()

# --- HELPER: Prediction Map ---
def get_prediction_map(image_stack, model, patch_size=64, stride=32):
    if model is None:
        return np.zeros((image_stack.shape[0], image_stack.shape[1]), dtype=np.uint8)
    h, w, _ = image_stack.shape
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    patches, coords = [], []
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image_stack[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    if not patches:
        return prediction_map
    patches = np.array(patches)
    predictions = model.predict(patches)
    predicted_classes = np.argmax(predictions, axis=1)
    for (y, x), predicted_class in zip(coords, predicted_classes):
        prediction_map[y:y+patch_size, x:x+patch_size] = predicted_class + 1
    return prediction_map

# --- SIDEBAR ---
st.sidebar.header("🌍 Select Farm Location")
locations = ["ludhiana", "ujjain", "thanjavur"]
selected_loc = st.sidebar.selectbox("Choose a farm:", locations)

# --- GLOBAL STATE ---
farm_results = {}

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["🌱 Crop Health", "🌍 Soil Condition", "🐛 Pest Risks", "📊 Summary"]
)

# ======================================================
# 🌱 CROP HEALTH TAB
# ======================================================
with tab1:
    st.header(f"🌱 Crop Health Monitoring – {selected_loc.title()}")

    if model:
        try:
            project_root = Path(".").resolve()
            processed_path = project_root / "data" / "processed" / selected_loc

            ndvi_path = list(processed_path.glob("*NDVI.tif"))[0]
            savi_path = list(processed_path.glob("*SAVI.tif"))[0]
            ndwi_path = list(processed_path.glob("*NDWI.tif"))[0]

            with rasterio.open(ndvi_path) as src: ndvi = src.read(1)
            with rasterio.open(savi_path) as src: savi = src.read(1)
            with rasterio.open(ndwi_path) as src: ndwi = src.read(1)

            # Normalize + stack
            ndvi_norm = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
            savi_norm = np.clip((savi + 1) / 2 * 255, 0, 255).astype(np.uint8)
            ndwi_norm = np.clip((ndwi + 1) / 2 * 255, 0, 255).astype(np.uint8)
            stacked_indices = np.stack([ndvi_norm, savi_norm, ndwi_norm], axis=-1)

            col1, col2 = st.columns(2)

            # --- Show NDVI Map ---
            with col1:
                st.subheader("🌱 Original NDVI Map")
                fig, ax = plt.subplots()
                im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, label="NDVI Value")
                ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig)

            # --- Show AI Prediction Map ---
            with col2:
                st.subheader("🤖 CropVision Health Map")
                with st.spinner("AI analyzing farm imagery..."):
                    prediction_map = get_prediction_map(stacked_indices, model)

                cmap = mcolors.ListedColormap(["red", "yellow", "green"])
                bounds = [0.5, 1.5, 2.5, 3.5]
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                fig, ax = plt.subplots()
                im = ax.imshow(prediction_map, cmap=cmap, norm=norm)
                cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3])
                cbar.ax.set_yticklabels(["High Stress", "Moderate Stress", "Healthy"])
                ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig)

                # --- Summary Stats ---
                total_pixels = np.count_nonzero(prediction_map)
                healthy_pixels = np.sum(prediction_map == 3)
                moderate_pixels = np.sum(prediction_map == 2)
                stress_pixels = np.sum(prediction_map == 1)

                if total_pixels > 0:
                    healthy_perc = (healthy_pixels / total_pixels) * 100
                    moderate_perc = (moderate_pixels / total_pixels) * 100
                    stress_perc = (stress_pixels / total_pixels) * 100

                    st.subheader("📊 Farm Health Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Healthy 🌿", f"{healthy_perc:.1f}%")
                    c2.metric("Moderate 🌾", f"{moderate_perc:.1f}%")
                    c3.metric("High Stress 🚨", f"{stress_perc:.1f}%")

                    farm_results[selected_loc] = {
                        "healthy": healthy_perc,
                        "moderate": moderate_perc,
                        "stress": stress_perc
                    }

        except Exception as e:
            st.error(f"⚠️ Error fetching data for {selected_loc}: {e}")

# ======================================================
# 🌍 SOIL TAB (Simulated per farm)
# ======================================================
with tab2:
    st.header(f"🌍 Soil Condition – {selected_loc.title()}")
    np.random.seed(hash(selected_loc) % 123456)
    soil_data = {
        "Moisture (%)": np.random.randint(20, 80, 10),
        "Nitrogen (N)": np.random.randint(30, 100, 10),
        "Phosphorus (P)": np.random.randint(20, 90, 10),
        "Potassium (K)": np.random.randint(25, 95, 10),
        "pH": np.random.uniform(5.5, 8.0, 10)
    }
    soil_df = pd.DataFrame(soil_data)
    st.dataframe(soil_df.style.background_gradient(cmap="YlGn"))

# ======================================================
# 🐛 PEST RISK TAB (Derived from stress %)
# ======================================================
# ======================================================
# 🐛 PEST RISK TAB (Stress + Weather Driven)
# ======================================================
with tab3:
    st.header(f"🐛 Pest Risk – {selected_loc.title()}")

    # Generate weather (or load real data later)
    weather_df = pd.DataFrame({
        "Day": pd.date_range("2025-09-28", periods=7, freq="D"),
        "Temp (°C)": np.random.randint(20, 35, 7),
        "Rainfall (mm)": np.random.randint(0, 15, 7),
        "Humidity (%)": np.random.randint(40, 90, 7)
    })

    avg_humidity = weather_df["Humidity (%)"].mean()
    avg_rain = weather_df["Rainfall (mm)"].mean()

    # Default
    pest_risk = "Unknown"
    explanation = ""

    if selected_loc in farm_results:
        stress = farm_results[selected_loc]["stress"]

        # Rule-based assessment
        if stress > 40 and avg_humidity > 70:
            pest_risk = "High"
            explanation = "🚨 High crop stress combined with humid weather increases pest outbreak risk."
        elif stress > 20 and (avg_humidity > 60 or avg_rain > 5):
            pest_risk = "Moderate"
            explanation = "⚠️ Moderate stress and favorable weather may allow pests to thrive."
        elif stress <= 20 and avg_humidity < 60:
            pest_risk = "Low"
            explanation = "✅ Healthy crops and dry conditions reduce pest risk."
        else:
            pest_risk = "Moderate"
            explanation = "⚠️ Conditions are mixed, moderate risk likely."

    # Display Metric
    st.metric("Pest Risk Level", pest_risk)

    # Show Pie Chart of Crop Health Distribution
    fig_pie = px.pie(
        names=["Healthy", "Moderate", "Stressed"],
        values=[
            farm_results.get(selected_loc, {}).get("healthy", 0),
            farm_results.get(selected_loc, {}).get("moderate", 0),
            farm_results.get(selected_loc, {}).get("stress", 0),
        ],
        color=["Healthy", "Moderate", "Stressed"],
        color_discrete_map={"Healthy": "green", "Moderate": "orange", "Stressed": "red"},
        title="Crop Health Composition"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Explanation
    if explanation:
        if pest_risk == "High":
            st.error(explanation)
        elif pest_risk == "Moderate":
            st.warning(explanation)
        else:
            st.success(explanation)


# ======================================================
# 📊 SUMMARY TAB (Farm-specific)
# ======================================================
with tab4:
    st.header(f"📊 Overall Summary – {selected_loc.title()}")
    if selected_loc in farm_results:
        healthy = farm_results[selected_loc]["healthy"]
        stress = farm_results[selected_loc]["stress"]

        if stress > 40:
            status = "🚨 Poor – Immediate action needed"
        elif stress > 20:
            status = "⚠️ Moderate – Monitor and intervene"
        else:
            status = "✅ Good – Farm is healthy"

        col1, col2, col3 = st.columns(3)
        col1.metric("Healthy 🌿", f"{healthy:.1f}%")
        col2.metric("Stressed 🚨", f"{stress:.1f}%")
        col3.metric("Overall Status", status)
    else:
        st.info("ℹ️ Please analyze Crop Health first to see summary.")
