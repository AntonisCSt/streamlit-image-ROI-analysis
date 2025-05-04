import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from PIL import Image, ImageDraw
import io
import base64
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Image ROI Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-16txtl3 h4 {
        margin-top: 0;
    }
    .roi-info {
        border-left: 3px solid #4e8df5;
        padding-left: 10px;
        margin-bottom: 15px;
    }
    .similarity-table {
        width: 100%;
        border-collapse: collapse;
    }
    .similarity-table th, .similarity-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .similarity-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .similarity-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        background-color: #4e8df5;
        color: white;
    }
    .high-similarity {
        background-color: #c6efce;
    }
    .medium-similarity {
        background-color: #ffeb9c;
    }
    .low-similarity {
        background-color: #ffc7ce;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate a unique color for each ROI
def get_unique_color(index):
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 0),    # Dark Green
        (128, 0, 0),    # Maroon
    ]
    return colors[index % len(colors)]

# Function to calculate histograms
def calculate_histogram(image_data, mask=None):
    # Split the channels
    b, g, r = cv2.split(image_data)
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Calculate histograms
    hist_r = cv2.calcHist([r], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], mask, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], mask, [256], [0, 256])
    
    hist_h = cv2.calcHist([h], [0], mask, [180], [0, 180])
    hist_s = cv2.calcHist([s], [0], mask, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [256], [0, 256])
    
    # Normalize for better visualization
    hist_r = cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
    hist_g = cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
    hist_b = cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    
    hist_h = cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    hist_s = cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    hist_v = cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
    
    return {
        'RGB': (hist_r, hist_g, hist_b),
        'HSV': (hist_h, hist_s, hist_v)
    }

# Function to calculate mean RGB and HSV values
def calculate_mean_values(image_data, mask=None):
    if mask is not None:
        # Ensure mask is binary (0 or 255)
        binary_mask = mask / 255.0
        
        # Calculate mean BGR values
        mean_b = np.mean(image_data[:, :, 0][mask > 0])
        mean_g = np.mean(image_data[:, :, 1][mask > 0])
        mean_r = np.mean(image_data[:, :, 2][mask > 0])
        
        # Convert to HSV and calculate mean
        hsv_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv_image[:, :, 0][mask > 0])
        mean_s = np.mean(hsv_image[:, :, 1][mask > 0])
        mean_v = np.mean(hsv_image[:, :, 2][mask > 0])
    else:
        # Calculate mean BGR values for the entire image
        mean_b = np.mean(image_data[:, :, 0])
        mean_g = np.mean(image_data[:, :, 1])
        mean_r = np.mean(image_data[:, :, 2])
        
        # Convert to HSV and calculate mean
        hsv_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv_image[:, :, 0])
        mean_s = np.mean(hsv_image[:, :, 1])
        mean_v = np.mean(hsv_image[:, :, 2])
    
    return {
        'RGB': (round(mean_r, 2), round(mean_g, 2), round(mean_b, 2)),
        'HSV': (round(mean_h, 2), round(mean_s, 2), round(mean_v, 2))
    }

# Function to calculate histogram similarity between two ROIs
def calculate_similarity(hist1, hist2):
    # Compute histogram correlation for each channel
    corr_r = cv2.compareHist(hist1['RGB'][0], hist2['RGB'][0], cv2.HISTCMP_CORREL)
    corr_g = cv2.compareHist(hist1['RGB'][1], hist2['RGB'][1], cv2.HISTCMP_CORREL)
    corr_b = cv2.compareHist(hist1['RGB'][2], hist2['RGB'][2], cv2.HISTCMP_CORREL)
    
    corr_h = cv2.compareHist(hist1['HSV'][0], hist2['HSV'][0], cv2.HISTCMP_CORREL)
    
    # Calculate average RGB correlation
    rgb_similarity = (corr_r + corr_g + corr_b) / 3
    
    # Return both RGB similarity and Hue similarity
    return {
        'RGB': round(rgb_similarity * 100, 2),
        'Hue': round(corr_h * 100, 2)
    }

# Create a mask for a rectangle
def create_rectangle_mask(image_shape, x1, y1, x2, y2):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

# Function to draw rectangles on image
def draw_rectangles_on_image(image, rectangles, labels=None):
    img_copy = image.copy()
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        color = get_unique_color(i)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label if provided
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img_copy, labels[i], (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img_copy

# Function to create histogram plots
def plot_histograms(histograms, color_name, color_code, title):
    # Increased size for better visibility
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(histograms[color_name], color=color_code)
    ax.set_title(f"{title} Histogram")
    ax.set_xlim([0, len(histograms[color_name])])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# Convert matplotlib figure to Streamlit compatible image
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# Function to create image from drawn rectangles for download
def create_roi_image(img, rectangles, labels):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        color = get_unique_color(i)
        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
        
        # Add label if provided
        if labels and i < len(labels) and labels[i]:
            draw.text((x1, y1-15), labels[i], fill=rgb_color)
    
    return pil_img

# Get similarity class for CSS styling
def get_similarity_class(similarity_value):
    if similarity_value >= 80:
        return "high-similarity"
    elif similarity_value >= 50:
        return "medium-similarity"
    else:
        return "low-similarity"

# Main function
def main():
    # App title
    st.title("🔍 Image ROI Analyzer")
    st.write("Upload an image, draw rectangles to analyze regions of interest, and save your selections!")

    # Initialize session state variables
    if 'rectangles' not in st.session_state:
        st.session_state.rectangles = []
    if 'labels' not in st.session_state:
        st.session_state.labels = []
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'histograms' not in st.session_state:
        st.session_state.histograms = []

    # Sidebar for image upload and rectangle control
    with st.sidebar:
        st.header("Controls")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            # Read the image and keep a copy of the original
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert RGB to BGR (OpenCV format)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # Handle grayscale or RGBA images
                if len(img_array.shape) == 2:
                    # Grayscale to BGR
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:
                    # RGBA to BGR
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            # Update session state with the new image
            if st.session_state.current_image is None or not np.array_equal(img_cv, st.session_state.current_image):
                st.session_state.last_image = st.session_state.current_image
                st.session_state.current_image = img_cv
                # Clear histograms when new image is loaded
                st.session_state.histograms = []
        
        # If we have an image, show rectangle controls
        if st.session_state.current_image is not None:
            st.subheader("Rectangle Controls")
            
            # Add a new rectangle
            with st.expander("Add New Rectangle", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    x1 = st.number_input("X1", 0, st.session_state.current_image.shape[1], 10, key="new_x1")
                    y1 = st.number_input("Y1", 0, st.session_state.current_image.shape[0], 10, key="new_y1")
                with col2:
                    x2 = st.number_input("X2", 0, st.session_state.current_image.shape[1], min(100, st.session_state.current_image.shape[1]-10), key="new_x2")
                    y2 = st.number_input("Y2", 0, st.session_state.current_image.shape[0], min(100, st.session_state.current_image.shape[0]-10), key="new_y2")
                
                label = st.text_input("Label (optional)", key="new_label")
                
                if st.button("Add Rectangle"):
                    # Make sure x1 < x2 and y1 < y2
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    st.session_state.rectangles.append((x1, y1, x2, y2))
                    st.session_state.labels.append(label)
                    
                    # Calculate histogram for the new rectangle
                    mask = create_rectangle_mask(st.session_state.current_image.shape, x1, y1, x2, y2)
                    hist = calculate_histogram(st.session_state.current_image, mask)
                    
                    # Add histogram to session state
                    st.session_state.histograms.append(hist)
                    
                    st.rerun()
            
            # Manage existing rectangles
            if st.session_state.rectangles:
                with st.expander("Manage Rectangles"):
                    for i, (rect, label) in enumerate(zip(st.session_state.rectangles, st.session_state.labels)):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"Rectangle {i+1}")
                        with col2:
                            st.write(f"Label: {label or 'None'}")
                        with col3:
                            if st.button("Remove", key=f"remove_{i}"):
                                st.session_state.rectangles.pop(i)
                                st.session_state.labels.pop(i)
                                if i < len(st.session_state.histograms):
                                    st.session_state.histograms.pop(i)
                                st.rerun()
            
            # Load/Save rectangle configurations
            st.subheader("Save/Load Configurations")
            
            # Save current configuration
            if st.session_state.rectangles:
                save_name = st.text_input("Configuration Name", 
                                         value=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                if st.button("Save Configuration"):
                    config = {
                        "rectangles": st.session_state.rectangles,
                        "labels": st.session_state.labels,
                        "image_shape": [st.session_state.current_image.shape[0], 
                                       st.session_state.current_image.shape[1]]
                    }
                    
                    # Convert to JSON string
                    config_json = json.dumps(config)
                    
                    # Create download link
                    b64 = base64.b64encode(config_json.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{save_name}.json">Download Configuration</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Load configuration
            st.write("Load Configuration:")
            config_file = st.file_uploader("Upload a configuration file", type=["json"])
            if config_file is not None:
                try:
                    config = json.load(config_file)
                    if st.button("Apply Configuration"):
                        # Check if configuration is compatible with current image
                        current_shape = [st.session_state.current_image.shape[0], 
                                        st.session_state.current_image.shape[1]]
                        config_shape = config.get("image_shape")
                        
                        if config_shape and (current_shape[0] != config_shape[0] or 
                                           current_shape[1] != config_shape[1]):
                            st.warning(f"⚠️ Original image dimensions ({config_shape[1]}x{config_shape[0]}) " +
                                      f"differ from current image ({current_shape[1]}x{current_shape[0]}). " +
                                      "Rectangles may need adjustment.")
                        
                        # Apply configuration
                        st.session_state.rectangles = config["rectangles"]
                        st.session_state.labels = config["labels"]
                        
                        # Recalculate histograms for all rectangles
                        st.session_state.histograms = []
                        for x1, y1, x2, y2 in st.session_state.rectangles:
                            mask = create_rectangle_mask(st.session_state.current_image.shape, x1, y1, x2, y2)
                            hist = calculate_histogram(st.session_state.current_image, mask)
                            st.session_state.histograms.append(hist)
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")
            
            # Clear all rectangles
            if st.session_state.rectangles:
                if st.button("Clear All Rectangles"):
                    st.session_state.rectangles = []
                    st.session_state.labels = []
                    st.session_state.histograms = []
                    st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Display the image with rectangles if available
        if st.session_state.current_image is not None:
            if st.session_state.rectangles:
                img_with_rects = draw_rectangles_on_image(
                    st.session_state.current_image, 
                    st.session_state.rectangles,
                    st.session_state.labels
                )
                st.image(cv2.cvtColor(img_with_rects, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Allow downloading the image with ROIs
                roi_img = create_roi_image(
                    st.session_state.current_image,
                    st.session_state.rectangles,
                    st.session_state.labels
                )
                
                # Convert PIL image to bytes
                roi_bytes = io.BytesIO()
                roi_img.save(roi_bytes, format='PNG')
                roi_bytes.seek(0)
                
                # Create download button
                st.download_button(
                    label="Download Image with ROIs",
                    data=roi_bytes,
                    file_name="image_with_rois.png",
                    mime="image/png"
                )
                
                # Display similarity matrix if we have more than one rectangle
                if len(st.session_state.rectangles) > 1:
                    st.subheader("ROI Similarity Analysis")
                    
                    # Calculate or update histogram data if needed
                    if len(st.session_state.histograms) != len(st.session_state.rectangles):
                        st.session_state.histograms = []
                        for x1, y1, x2, y2 in st.session_state.rectangles:
                            mask = create_rectangle_mask(st.session_state.current_image.shape, x1, y1, x2, y2)
                            hist = calculate_histogram(st.session_state.current_image, mask)
                            st.session_state.histograms.append(hist)
                    
                    # Select similarity type
                    similarity_type = st.radio(
                        "Similarity Metric:",
                        ["RGB Similarity", "Hue Similarity"],
                        horizontal=True
                    )
                    
                    # Create similarity matrix
                    num_rois = len(st.session_state.rectangles)
                    similarity_matrix = np.zeros((num_rois, num_rois))
                    rgb_similarity_matrix = np.zeros((num_rois, num_rois))
                    hue_similarity_matrix = np.zeros((num_rois, num_rois))
                    
                    for i in range(num_rois):
                        for j in range(num_rois):
                            if i == j:
                                rgb_similarity_matrix[i, j] = 100.0  # Same ROI has 100% similarity
                                hue_similarity_matrix[i, j] = 100.0  # Same ROI has 100% similarity
                            else:
                                sim = calculate_similarity(st.session_state.histograms[i], st.session_state.histograms[j])
                                rgb_similarity_matrix[i, j] = sim['RGB']
                                hue_similarity_matrix[i, j] = sim['Hue']
                    
                    # Set the displayed similarity matrix based on user selection
                    if similarity_type == "RGB Similarity":
                        similarity_matrix = rgb_similarity_matrix
                    else:
                        similarity_matrix = hue_similarity_matrix
                    
                    # Calculate average similarities for each ROI
                    rgb_avg_similarities = []
                    hue_avg_similarities = []
                    
                    for i in range(num_rois):
                        # Calculate average similarity with all other ROIs (excluding self)
                        other_indices = [j for j in range(num_rois) if j != i]
                        if other_indices:  # Check if there are other ROIs to compare with
                            rgb_avg = np.mean([rgb_similarity_matrix[i, j] for j in other_indices])
                            hue_avg = np.mean([hue_similarity_matrix[i, j] for j in other_indices])
                        else:
                            rgb_avg = 0
                            hue_avg = 0
                        
                        rgb_avg_similarities.append(rgb_avg)
                        hue_avg_similarities.append(hue_avg)
                    
                    # Calculate overall average similarities
                    overall_rgb_avg = np.mean(rgb_avg_similarities) if rgb_avg_similarities else 0
                    overall_hue_avg = np.mean(hue_avg_similarities) if hue_avg_similarities else 0
                    
                    # Create a visually appealing similarity table
                    roi_names = [f"ROI {i+1}: {label or 'Unlabeled'}" for i, label in enumerate(st.session_state.labels)]
                    
                    # Create HTML table for better formatting
                    html_table = "<table class='similarity-table'><tr><th></th>"
                    for name in roi_names:
                        html_table += f"<th>{name}</th>"
                    html_table += "<th>Avg RGB Sim</th><th>Avg Hue Sim</th></tr>"
                    
                    for i, row_name in enumerate(roi_names):
                        html_table += f"<tr><th>{row_name}</th>"
                        for j in range(num_rois):
                            similarity_value = similarity_matrix[i, j]
                            similarity_class = get_similarity_class(similarity_value)
                            html_table += f"<td class='{similarity_class}'>{similarity_value:.1f}%</td>"
                        
                        # Add average similarity columns
                        rgb_avg_class = get_similarity_class(rgb_avg_similarities[i])
                        hue_avg_class = get_similarity_class(hue_avg_similarities[i])
                        html_table += f"<td class='{rgb_avg_class}'>{rgb_avg_similarities[i]:.1f}%</td>"
                        html_table += f"<td class='{hue_avg_class}'>{hue_avg_similarities[i]:.1f}%</td>"
                        html_table += "</tr>"
                    
                    # Add overall average row
                    html_table += "<tr><th>Overall Average</th>"
                    for _ in range(num_rois):
                        html_table += "<td>-</td>"
                    
                    overall_rgb_class = get_similarity_class(overall_rgb_avg)
                    overall_hue_class = get_similarity_class(overall_hue_avg)
                    html_table += f"<td class='{overall_rgb_class}'><strong>{overall_rgb_avg:.1f}%</strong></td>"
                    html_table += f"<td class='{overall_hue_class}'><strong>{overall_hue_avg:.1f}%</strong></td>"
                    html_table += "</tr></table>"
                    
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Add CSS for the table styling
                    st.markdown("""
                    <style>
                    .similarity-table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    .similarity-table th, .similarity-table td {
                        padding: 8px;
                        text-align: center;
                        border: 1px solid #ddd;
                    }
                    .similarity-table th {
                        background-color: #f2f2f2;
                    }
                    .high-similarity {
                        background-color: #c8e6c9;
                    }
                    .medium-similarity {
                        background-color: #fff9c4;
                    }
                    .low-similarity {
                        background-color: #ffcdd2;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Add a legend for the color coding
                    legend_col1, legend_col2, legend_col3 = st.columns(3)
                    with legend_col1:
                        st.markdown("<div class='high-similarity' style='padding:5px;'>High Similarity (≥80%)</div>", unsafe_allow_html=True)
                    with legend_col2:
                        st.markdown("<div class='medium-similarity' style='padding:5px;'>Medium Similarity (50-79%)</div>", unsafe_allow_html=True)
                    with legend_col3:
                        st.markdown("<div class='low-similarity' style='padding:5px;'>Low Similarity (<50%)</div>", unsafe_allow_html=True)                    
            else:
                st.image(cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.info("Draw rectangles using the controls in the sidebar to analyze regions of interest.")
        else:
            st.info("Please upload an image using the sidebar to get started.")
    
    with col2:
        with st.expander("Show ROI Analysis"):
            # Display analysis if we have rectangles and an image
            if st.session_state.current_image is not None and st.session_state.rectangles:
                st.subheader("ROI Analysis")
                
                # Select ROI to display
                roi_options = [f"ROI {i+1}: {label or 'Unlabeled'}" for i, label in enumerate(st.session_state.labels)]
                selected_roi = st.selectbox("Select ROI to analyze", roi_options)
                roi_index = int(selected_roi.split(":")[0].replace("ROI ", "")) - 1
                
                # Get the selected rectangle
                x1, y1, x2, y2 = st.session_state.rectangles[roi_index]
                
                # Extract the ROI from the image
                roi = st.session_state.current_image[y1:y2, x1:x2]
                
                # Display ROI
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption=f"Selected ROI: {st.session_state.labels[roi_index] or 'Unlabeled'}",width=50)
                
                # Create mask for the ROI
                mask = create_rectangle_mask(st.session_state.current_image.shape, x1, y1, x2, y2)
                
                # Calculate mean values
                mean_values = calculate_mean_values(st.session_state.current_image, mask)
                
                # Display mean values
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.subheader("Mean Color Values")
                
                col_r, col_g, col_b = st.columns(3)
                with col_r:
                    st.markdown(f"**R:** {mean_values['RGB'][0]}")
                with col_g:
                    st.markdown(f"**G:** {mean_values['RGB'][1]}")
                with col_b:
                    st.markdown(f"**B:** {mean_values['RGB'][2]}")
                
                col_h, col_s, col_v = st.columns(3)
                with col_h:
                    st.markdown(f"**H:** {mean_values['HSV'][0]}")
                with col_s:
                    st.markdown(f"**S:** {mean_values['HSV'][1]}")
                with col_v:
                    st.markdown(f"**V:** {mean_values['HSV'][2]}")
                
                # Show color swatch
                r, g, b = [int(x) for x in mean_values['RGB']]
                color_swatch = f"<div style='background-color: rgb({r},{g},{b}); width:100%; height:30px; border-radius:5px;'></div>"
                st.markdown(color_swatch, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Calculate histograms if needed or use existing ones
                if roi_index < len(st.session_state.histograms):
                    histograms = st.session_state.histograms[roi_index]
                else:
                    histograms = calculate_histogram(st.session_state.current_image, mask)
                    st.session_state.histograms.append(histograms)
                
                # Display histograms with increased size
                st.subheader("RGB Histograms")
                
                # RGB histograms - using full width for better visibility
                fig_r = plot_histograms(histograms['RGB'], 0, 'red', 'Red')
                st.image(fig_to_image(fig_r))
                
                fig_g = plot_histograms(histograms['RGB'], 1, 'green', 'Green')
                st.image(fig_to_image(fig_g))
                
                fig_b = plot_histograms(histograms['RGB'], 2, 'blue', 'Blue')
                st.image(fig_to_image(fig_b))
                
                # HSV histograms
                st.subheader("HSV Histograms")
                
                fig_h = plot_histograms(histograms['HSV'], 0, 'orange', 'Hue')
                st.image(fig_to_image(fig_h))
                
                fig_s = plot_histograms(histograms['HSV'], 1, 'purple', 'Saturation')
                st.image(fig_to_image(fig_s))
                
                fig_v = plot_histograms(histograms['HSV'], 2, 'gray', 'Value')
                st.image(fig_to_image(fig_v))

if __name__ == "__main__":
    main()