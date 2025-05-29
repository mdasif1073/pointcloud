import asyncio
import nest_asyncio
import streamlit as st
from streamlit.web.bootstrap import run
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import open3d as o3d
import torch.nn.functional as F
import os
from io import BytesIO
import gc

# Fix event loop and watcher issues
try:
    nest_asyncio.apply()
except RuntimeError:
    pass
run._has_watchdog = False

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory optimization configurations
MAX_POINTS = 200000
DEFAULT_BATCH_SIZE = 512
IMAGE_SIZE = 128

# Set page config
st.set_page_config(page_title="Point Cloud Style Transfer", layout="wide")

# App title
st.title("🎨 Point Cloud Style Transfer")
st.markdown("""
Transfer artistic styles from images to 3D point clouds using deep learning.
Upload a point cloud (PLY format) and a style image to begin.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    uploaded_pc = st.file_uploader("Upload Point Cloud (PLY)", type=["ply"])
    uploaded_style = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    
    st.subheader("Processing Options")
    target_points = st.slider("Max Points", 10000, MAX_POINTS, 50000, 10000)
    batch_size = st.selectbox("Batch Size", [256, 512, 1024], index=1)
    
    st.subheader("Memory Options")
    aggressive_cleanup = st.checkbox("Aggressive Memory Cleanup", value=True)
    
    st.subheader("Model Options")
    style_layers = st.multiselect(
        "Style Layers for Model Input",
        ['relu1_1', 'relu2_1', 'relu3_1'],
        ['relu1_1', 'relu2_1']
    )

# --- Model Definitions ---
class LiteVGGStyleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList([
            vgg[:3],   # relu1_1
            vgg[3:8],  # relu2_1
            vgg[8:17]  # relu3_1
        ])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, slice in enumerate(self.slices):
            x = slice(x)
            if f'relu{i+1}_1' in style_layers:
                features.append(x)
        return features

class PointCloudStyleTransferLite(nn.Module):
    def __init__(self, style_dims, color_proj_dims, geom_dim=64):
        super().__init__()
        self.geom_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        total_style_dim = style_dims[0]
        
        self.mlp = nn.Sequential(
            nn.Linear(64 + total_style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
        self.color_projectors = nn.ModuleList([
            nn.Linear(3, dim) for dim in color_proj_dims
        ])

    def forward(self, points, style_feat_concat):
        geom = self.geom_encoder(points)
        inp = torch.cat([geom, style_feat_concat], dim=1)
        colors = self.mlp(inp)
        
        projected_features = []
        for proj in self.color_projectors:
            projected_features.append(proj(colors).unsqueeze(1))
            
        return colors, projected_features

# --- Optimized Processing Functions ---
def load_point_cloud(uploaded_file, target_points=None):
    """Memory-optimized point cloud loading"""
    temp_path = "temp_pointcloud.ply"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pcd = o3d.io.read_point_cloud(temp_path)
        
        if target_points and len(pcd.points) > target_points:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            if len(pcd.points) > target_points:
                pcd = pcd.random_down_sample(target_points / len(pcd.points))
        
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        
        return points, colors
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if aggressive_cleanup:
            gc.collect()

def load_style_image(uploaded_file, device, imsize=IMAGE_SIZE):
    """Memory-optimized style image loading"""
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(uploaded_file).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    return img

def gram_matrix(y):
    b, ch, *rest = y.size()
    features = y.view(b, ch, -1)
    if features.size(-1) == 0:
        return torch.zeros(b, ch, ch, device=y.device, dtype=y.dtype)
    gram = torch.bmm(features, features.transpose(1, 2)) / features.size(-1)
    return gram

def process_in_batches(points, model, style_feat_concat, batch_size=DEFAULT_BATCH_SIZE):
    """Process data in memory-friendly batches"""
    results = []
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    for i in range(0, len(points_tensor), batch_size):
        # Process batch
        batch_points = points_tensor[i:i + batch_size]
        batch_style = style_feat_concat[i:i + batch_size]
        
        with torch.no_grad():
            colors, _ = model(batch_points, batch_style)
            results.append(colors.cpu().numpy())
        
        # Cleanup
        del batch_points, batch_style, colors
        if aggressive_cleanup:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final cleanup
    del points_tensor
    if aggressive_cleanup:
        torch.cuda.empty_cache()
    return np.concatenate(results, axis=0)

def visualize_point_cloud(points, colors=None, title="Point Cloud"):
    """Memory-efficient visualization"""
    fig = plt.figure(figsize=(5, 5))
    if colors is None:
        plt.scatter(points[:, 0], points[:, 1], s=0.5)
    else:
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=0.5)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    return fig

# --- Load Models ---
@st.cache_resource
def load_models():
    style_extractor = LiteVGGStyleExtractor().to(device)
    model = PointCloudStyleTransferLite(
        style_dims=[1],  # Will be updated
        color_proj_dims=[64, 128, 256]  # Reduced channel dimensions
    ).to(device)
    return style_extractor, model

# --- Main App Logic ---
if uploaded_pc and uploaded_style:
    try:
        # Initialize with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load models
        status_text.text("Loading models...")
        style_extractor, model = load_models()
        progress_bar.progress(10)
        
        # Load inputs
        status_text.text("Loading inputs...")
        points_np, original_colors = load_point_cloud(uploaded_pc, min(target_points, MAX_POINTS))
        style_img = load_style_image(uploaded_style, device)
        progress_bar.progress(30)
        
        # Show previews
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Point Cloud")
            fig = visualize_point_cloud(points_np, original_colors)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.subheader("Style Image")
            st.image(uploaded_style, use_container_width=True)
        
        progress_bar.progress(40)
        
        # Style transfer
        status_text.text("Extracting style features...")
        with torch.no_grad():
            features = style_extractor(style_img)
            
            style_grams = []
            channel_dims = []
            for i, layer_name in enumerate(['relu1_1', 'relu2_1', 'relu3_1']):
                if layer_name in style_layers:
                    gram = gram_matrix(features[i])
                    style_grams.append(gram.view(-1))
                    channel_dims.append(features[i].shape[1])
            
            total_style_dim = sum(g.numel() for g in style_grams)
            style_concat = torch.cat(style_grams).unsqueeze(0).repeat(len(points_np), 1)
            
            # Update model
            model = PointCloudStyleTransferLite(
                style_dims=[total_style_dim],
                color_proj_dims=channel_dims
            ).to(device)
            
            # Load weights would go here
            # model.load_state_dict(torch.load("model.pth", map_location=device))
            
            # Cleanup
            del features, style_grams
            if aggressive_cleanup:
                torch.cuda.empty_cache()
        
        progress_bar.progress(60)
        
        # Process point cloud
        status_text.text("Applying style transfer...")
        stylized_colors = process_in_batches(points_np, model, style_concat, batch_size)
        progress_bar.progress(90)
        
        # Show results
        st.subheader("Stylized Point Cloud")
        fig = visualize_point_cloud(points_np, stylized_colors)
        st.pyplot(fig)
        plt.close(fig)
        
        # Create download
        temp_output = "temp_styled.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(stylized_colors)
        o3d.io.write_point_cloud(temp_output, pcd)
        
        with open(temp_output, "rb") as f:
            st.download_button(
                "Download Styled Point Cloud",
                f,
                file_name="styled_output.ply",
                mime="application/octet-stream"
            )
        
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        progress_bar.progress(100)
        status_text.text("Style transfer complete!")
        st.success("Done!")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if "memory" in str(e).lower():
            st.warning("""
            **Memory Error**: Try reducing:
            - Maximum number of points
            - Batch size
            - Number of style layers
            - Enable aggressive memory cleanup
            """)
        
        # Cleanup
        if 'points_np' in locals():
            del points_np
        if 'stylized_colors' in locals():
            del stylized_colors
        torch.cuda.empty_cache()
        gc.collect()
else:
    st.warning("Please upload both a point cloud and a style image to proceed.")

# Documentation
st.markdown("""
### Memory Optimization Tips:
1. Start with smaller point clouds (<50,000 points)
2. Use smaller batch sizes (256 or 512)
3. Select fewer style layers (1-2)
4. Enable "Aggressive Memory Cleanup"
5. Close other memory-intensive applications

### Recommended Workflow:
1. Test with small inputs first
2. Gradually increase complexity
3. Monitor system memory usage
""")