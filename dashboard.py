# dashboard.py
import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")
st.title("🌾 AI-Powered Crop Health Monitoring")

# --- LOAD THE MODEL (runs only once) ---
@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model('model/crop_health_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_keras_model()

# --- HELPER FUNCTIONS ---
def get_prediction_map(image_stack, model, patch_size=64, stride=32):
    """Extracts patches, predicts, and reconstructs the map."""
    if model is None:
        return np.zeros((image_stack.shape[0], image_stack.shape[1]), dtype=np.uint8)

    h, w, _ = image_stack.shape
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    
    patches = []
    coords = []
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
        prediction_map[y:y+patch_size, x:x+patch_size] = predicted_class + 1 # Use 1,2,3 for viz
        
    return prediction_map

# --- USER INTERFACE ---
st.sidebar.header("Select Farm Location")
locations = ['ludhiana', 'ujjain', 'thanjavur']
selected_loc = st.sidebar.selectbox("Choose a farm:", locations)

if model:
    try:
        # Load the data for the selected location
        project_root = Path('.').resolve()
        processed_path = project_root / 'data' / 'processed' / selected_loc

        st.info(f"Searching for data in: {processed_path}")
        
        ndvi_path = list(processed_path.glob('*NDVI.tif'))[0]
        savi_path = list(processed_path.glob('*SAVI.tif'))[0]
        ndwi_path = list(processed_path.glob('*NDWI.tif'))[0]

        with rasterio.open(ndvi_path) as src: ndvi = src.read(1)
        with rasterio.open(savi_path) as src: savi = src.read(1)
        with rasterio.open(ndwi_path) as src: ndwi = src.read(1)
        
        # Normalize and stack for prediction
        ndvi_norm = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
        savi_norm = np.clip((savi + 1) / 2 * 255, 0, 255).astype(np.uint8)
        ndwi_norm = np.clip((ndwi + 1) / 2 * 255, 0, 255).astype(np.uint8)
        stacked_indices = np.stack([ndvi_norm, savi_norm, ndwi_norm], axis=-1)

        # --- Display Original NDVI and Final Prediction ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original NDVI Map")
            fig, ax = plt.subplots()
            im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="NDVI Value")
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)

        with col2:
            st.subheader("AI-Generated Health Map")
            with st.spinner('AI is analyzing the farm...'):
                prediction_map = get_prediction_map(stacked_indices, model)
            
            # Custom colormap
            cmap = mcolors.ListedColormap(['red', 'yellow', 'green'])
            bounds = [0.5, 1.5, 2.5, 3.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots()
            im = ax.imshow(prediction_map, cmap=cmap, norm=norm)
            
            # Color bar with custom labels
            cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3])
            cbar.ax.set_yticklabels(['High Stress', 'Moderate Stress', 'Healthy'])
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)
            # --- (Inside the `with col2:` block, after generating the map) ---

            # Calculate summary statistics
            total_pixels = np.count_nonzero(prediction_map) # Count only non-zero pixels
            healthy_pixels = np.sum(prediction_map == 3)
            moderate_pixels = np.sum(prediction_map == 2)
            high_stress_pixels = np.sum(prediction_map == 1)

            if total_pixels > 0:
                healthy_perc = (healthy_pixels / total_pixels) * 100
                moderate_perc = (moderate_pixels / total_pixels) * 100
                high_stress_perc = (high_stress_pixels / total_pixels) * 100
    
                st.subheader("Farm Health Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Healthy", f"{healthy_perc:.1f}%")
                c2.metric("Moderate Stress", f"{moderate_perc:.1f}%")
                c3.metric("High Stress", f"{high_stress_perc:.1f}%")
            
    except IndexError:
        st.error(f"Data not found for {selected_loc}. Please make sure the processed .tif files exist.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add this somewhere on your dashboard
with st.expander("ℹ️ About This App"):
    st.write("""
        This dashboard uses Sentinel-2 satellite imagery to monitor crop health.
        - **NDVI Map:** Shows the raw vegetation index, a measure of plant greenness.
        - **AI-Generated Health Map:** A pre-trained MobileNetV2 model classifies the farm into three health categories (High Stress, Moderate, Healthy) based on a combination of NDVI, SAVI, and NDWI indices.
    """)