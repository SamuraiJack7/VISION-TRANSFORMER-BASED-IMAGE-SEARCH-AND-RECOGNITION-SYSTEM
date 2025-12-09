"""
Image Understanding UI - Standalone Version
Supports DeiT-Tiny and MobileViT-XXS models
Works with model files only - no dataset required!
"""

import gradio as gr
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Text-based retrieval using TF-IDF (no transformers needed!)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Retrieval is always available now!
RETRIEVAL_AVAILABLE = True
print("[OK] Text-based retrieval enabled using TF-IDF (no transformers needed!)")

# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model paths
MODEL_PATHS = {
    'deit_tiny': 'best_model1_deit_tiny.pth',
    'mobilevit_xxs': 'best_model1_mobilevit_xxs.pth'
}

# ============================================================================
# HARDCODED CLASSES AND ATTRIBUTES (from training)
# ============================================================================

# Class names (10 classes)
CLASS_NAMES = [
    'clothing_cap',
    'clothing_wrist_watch',
    'food_storage_plastic_container',
    'footwear_sneakers',
    'personal_care_deodorant',
    'personal_care_shampoo_bottle',
    'personal_care_soap_bar',
    'tableware_water_bottle',
    'travel_backpack',
    'travel_handbag'
]

# Color attributes (15 colors) - alphabetically sorted as LabelEncoder does
COLOR_NAMES = [
    'black', 'blue', 'brown', 'gold', 'gray', 'green', 'grey',
    'multicolor', 'pink', 'purple', 'red', 'silver', 'transparent',
    'white', 'yellow'
]

# Material attributes (9 materials) - alphabetically sorted as LabelEncoder does
MATERIAL_NAMES = [
    'cotton', 'fabric', 'glass', 'leather', 'metal',
    'plastic', 'rubber', 'soap', 'synthetic'
]

# Condition attributes (4 conditions) - alphabetically sorted as LabelEncoder does
CONDITION_NAMES = ['mint', 'new', 'used', 'worn']

num_classes = len(CLASS_NAMES)
num_colors = len(COLOR_NAMES)
num_materials = len(MATERIAL_NAMES)
num_conditions = len(CONDITION_NAMES)

print(f"Loaded: {num_classes} classes, {num_colors} colors, {num_materials} materials, {num_conditions} conditions")

# Image transforms
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class MultiTaskViT(nn.Module):
    """Multi-task Vision Transformer"""
    def __init__(self, model_name='mobilevit_xxs', num_classes=10, 
                 num_colors=10, num_materials=10, num_conditions=5, pretrained=False):
        super(MultiTaskViT, self).__init__()
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,
            global_pool='token' if 'deit' in model_name else 'avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        hidden_dim = feature_dim // 2
        
        # Task heads
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.color_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_colors)
        )
        self.material_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_materials)
        )
        self.condition_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_conditions)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = self.dropout(features)
        
        if return_features:
            return {
                'features': features,
                'class_logits': self.class_head(features),
                'color_logits': self.color_head(features),
                'material_logits': self.material_head(features),
                'condition_logits': self.condition_head(features)
            }
        
        return {
            'class_logits': self.class_head(features),
            'color_logits': self.color_head(features),
            'material_logits': self.material_head(features),
            'condition_logits': self.condition_head(features)
        }

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_type='mobilevit_xxs'):
    """Load a trained model"""
    model_names = {
        'deit_tiny': 'deit_tiny_patch16_224',
        'mobilevit_xxs': 'mobilevit_xxs'
    }
    
    try:
        print(f"\n[INFO] Loading {model_type} model...")
        
        model = MultiTaskViT(
            model_name=model_names[model_type],
            num_classes=num_classes,
            num_colors=num_colors,
            num_materials=num_materials,
            num_conditions=num_conditions,
            pretrained=False
        )
        
        checkpoint_path = MODEL_PATHS[model_type]
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Model file not found: {checkpoint_path}")
            return None
        
        print(f"[INFO] Loading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"[OK] Successfully loaded {model_type} model!")
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load {model_type} model: {str(e)}")
        return None

# Load both models
print("\n" + "="*70)
print("LOADING MODELS")
print("="*70)
models = {}
for model_type in ['deit_tiny', 'mobilevit_xxs']:
    models[model_type] = load_model(model_type)

print("\n[INFO] Model loading complete:")
for model_type, model in models.items():
    if model is not None:
        print(f"  [OK] {model_type}")
    else:
        print(f"  [FAILED] {model_type}")

# Global storage for uploaded gallery images, features, and text descriptions
gallery_images = []  # List of PIL Images
gallery_features = {}  # {model_type: numpy array of features}
gallery_descriptions = {}  # {model_type: List of text descriptions for each image}
tfidf_vectorizer = {}  # {model_type: TF-IDF vectorizer for text search}
tfidf_matrix = {}  # {model_type: TF-IDF matrix of gallery descriptions}

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_image(image, model_type='mobilevit_xxs'):
    """
    Predict class and attributes for an uploaded image
    
    Args:
        image: PIL Image
        model_type: 'deit_tiny' or 'mobilevit_xxs'
    
    Returns:
        HTML formatted results
    """
    if image is None:
        return "<p style='color: red;'>Please upload an image.</p>"
    
    model = models.get(model_type)
    if model is None:
        return f"<p style='color: red;'>Model {model_type} not available.</p>"
    
    # Preprocess image
    img_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Get predictions
        class_logits = outputs['class_logits']
        class_probs = torch.nn.functional.softmax(class_logits, dim=1)
        class_conf, class_idx = torch.max(class_probs, 1)
        
        color_logits = outputs['color_logits']
        color_probs = torch.nn.functional.softmax(color_logits, dim=1)
        color_conf, color_idx = torch.max(color_probs, 1)
        
        material_logits = outputs['material_logits']
        material_probs = torch.nn.functional.softmax(material_logits, dim=1)
        material_conf, material_idx = torch.max(material_probs, 1)
        
        condition_logits = outputs['condition_logits']
        condition_probs = torch.nn.functional.softmax(condition_logits, dim=1)
        condition_conf, condition_idx = torch.max(condition_probs, 1)
    
    # Decode predictions using hardcoded class names
    class_name = CLASS_NAMES[class_idx.item()]
    color_name = COLOR_NAMES[color_idx.item()]
    material_name = MATERIAL_NAMES[material_idx.item()]
    condition_name = CONDITION_NAMES[condition_idx.item()]
    
    # Format output as HTML
    model_display = "DeiT-Tiny" if model_type == 'deit_tiny' else "MobileViT-XXS"
    
    html_output = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h2 style="margin-top: 0; text-align: center;">🎯 Prediction Results</h2>
        <p style="text-align: center; opacity: 0.9; margin-bottom: 20px;"><b>Model:</b> {model_display}</p>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h3 style="margin-top: 0;">📦 Object Class</h3>
            <p style="font-size: 20px; margin: 10px 0;"><b>{class_name.replace('_', ' ').title()}</b></p>
            <p style="opacity: 0.9;">Confidence: {class_conf.item()*100:.2f}%</p>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h3 style="margin-top: 0;">🎨 Visual Attributes</h3>
            <p><b>Color:</b> {color_name.capitalize()} ({color_conf.item()*100:.1f}%)</p>
            <p><b>Material:</b> {material_name.capitalize()} ({material_conf.item()*100:.1f}%)</p>
            <p><b>Condition:</b> {condition_name.capitalize()} ({condition_conf.item()*100:.1f}%)</p>
        </div>
    </div>
    """
    
    return html_output

# ============================================================================
# IMAGE GALLERY & RETRIEVAL FUNCTIONS
# ============================================================================

def load_images_from_folder(folder_path='images', model_type='mobilevit_xxs', show_progress=False):
    """
    Load images from a local folder for gallery
    
    Args:
        folder_path: Path to folder containing images
        model_type: Model to use for feature extraction
        show_progress: Whether to show progress bar in console
    
    Returns:
        Status message or number of images loaded
    """
    global gallery_images, gallery_features, gallery_descriptions, tfidf_vectorizer, tfidf_matrix
    
    if not os.path.exists(folder_path):
        if show_progress:
            print(f"[WARNING] Folder '{folder_path}' not found. Skipping auto-load.")
            return 0
        return f"❌ Folder '{folder_path}' not found. Please create it and add images."
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) == 0:
        if show_progress:
            print(f"[INFO] No images found in '{folder_path}' folder. Skipping auto-load.")
            return 0
        return f"⚠️ No images found in '{folder_path}' folder."
    
    model = models.get(model_type)
    if model is None:
        if show_progress:
            print(f"[ERROR] Model {model_type} not available.")
            return 0
        return f"❌ Model {model_type} not available."
    
    # Load images only if not already loaded
    if len(gallery_images) == 0:
        if show_progress:
            print(f"[OK] Loading images from '{folder_path}' folder...")
            image_files_iter = tqdm(image_files, desc="Loading images", unit="img")
        else:
            image_files_iter = image_files
        
        for img_path in image_files_iter:
            try:
                img = Image.open(img_path).convert('RGB')
                gallery_images.append(img)
            except Exception as e:
                if not show_progress:
                    print(f"Error loading {img_path}: {e}")
                continue
        
        if len(gallery_images) == 0:
            if show_progress:
                print("[ERROR] Failed to load any images from folder.")
                return 0
            return "❌ Failed to load any images from folder."
    else:
        if show_progress:
            print(f"[OK] Reusing {len(gallery_images)} already loaded images...")
    
    # Extract features and predict attributes for each image
    if show_progress:
        print(f"[OK] Analyzing {len(gallery_images)} images with {model_type} model...")
    
    model.eval()
    features_list = []
    descriptions = []
    
    gallery_iter = tqdm(gallery_images, desc="Analyzing images", unit="img") if show_progress else gallery_images
    
    # Always generate descriptions for EACH model (so they give different results)
    with torch.no_grad():
        for img in gallery_iter:
            img_tensor = inference_transform(img).unsqueeze(0).to(device)
            
            # Extract visual features
            features = model.backbone(img_tensor)
            features_list.append(features.cpu().numpy())
            
            # Predict attributes to create text description for THIS model
            outputs = model(img_tensor)
            
            # Get predictions
            _, class_idx = torch.max(outputs['class_logits'], 1)
            _, color_idx = torch.max(outputs['color_logits'], 1)
            _, material_idx = torch.max(outputs['material_logits'], 1)
            _, condition_idx = torch.max(outputs['condition_logits'], 1)
            
            # Create text description
            class_name = CLASS_NAMES[class_idx.item()]
            color_name = COLOR_NAMES[color_idx.item()]
            material_name = MATERIAL_NAMES[material_idx.item()]
            condition_name = CONDITION_NAMES[condition_idx.item()]
            
            # Build searchable text description for THIS model
            description = f"{color_name} {material_name} {class_name} {condition_name}"
            descriptions.append(description)
    
    # Store features for this model
    gallery_features[model_type] = np.vstack(features_list)
    
    # Store descriptions for THIS model
    gallery_descriptions[model_type] = descriptions
    
    # Build TF-IDF vectorizer for THIS model
    if show_progress:
        print(f"[OK] Building TF-IDF index for {model_type}...")
    
    tfidf_vectorizer[model_type] = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix[model_type] = tfidf_vectorizer[model_type].fit_transform(descriptions)
    
    if show_progress:
        print(f"[OK] Features extracted for {model_type}! {len(gallery_images)} images ready.\n")
        return len(gallery_images)
    
    return f"✅ Successfully loaded {len(gallery_images)} images from '{folder_path}' folder!\n\nGallery is ready for text-based search.\nSearch using color, material, object type, or condition!"

def upload_gallery_images(images, model_type='mobilevit_xxs'):
    """
    Upload multiple images to create a searchable gallery using TF-IDF text search
    
    Args:
        images: List of uploaded images
        model_type: Model to use for feature extraction
    
    Returns:
        Status message
    """
    global gallery_images, gallery_features, gallery_descriptions, tfidf_vectorizer, tfidf_matrix
    
    if images is None or len(images) == 0:
        return "⚠️ No images uploaded. Please upload images first."
    
    model = models.get(model_type)
    if model is None:
        return f"❌ Model {model_type} not available."
    
    # Store images
    gallery_images = [Image.open(img).convert('RGB') if isinstance(img, str) else img for img in images]
    
    # Extract features and predict attributes for each image
    print(f"Extracting features and predicting attributes for {len(gallery_images)} images...")
    model.eval()
    features_list = []
    descriptions = []
    
    with torch.no_grad():
        for img in gallery_images:
            img_tensor = inference_transform(img).unsqueeze(0).to(device)
            
            # Extract visual features
            features = model.backbone(img_tensor)
            features_list.append(features.cpu().numpy())
            
            # Predict attributes to create text description
            outputs = model(img_tensor)
            
            # Get predictions
            _, class_idx = torch.max(outputs['class_logits'], 1)
            _, color_idx = torch.max(outputs['color_logits'], 1)
            _, material_idx = torch.max(outputs['material_logits'], 1)
            _, condition_idx = torch.max(outputs['condition_logits'], 1)
            
            # Create text description
            class_name = CLASS_NAMES[class_idx.item()]
            color_name = COLOR_NAMES[color_idx.item()]
            material_name = MATERIAL_NAMES[material_idx.item()]
            condition_name = CONDITION_NAMES[condition_idx.item()]
            
            # Build searchable text description
            description = f"{color_name} {material_name} {class_name} {condition_name}"
            descriptions.append(description)
    
    # Store features
    gallery_features[model_type] = np.vstack(features_list)
    gallery_descriptions[model_type] = descriptions
    
    # Build TF-IDF vectorizer for this model
    print(f"Building TF-IDF index for {model_type}...")
    tfidf_vectorizer[model_type] = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix[model_type] = tfidf_vectorizer[model_type].fit_transform(descriptions)
    
    return f"✅ Successfully uploaded {len(gallery_images)} images!\n\nGallery ready for {model_type}. Search using color, material, object type, or condition!"

def search_images(query_text, model_type='mobilevit_xxs', top_k=6):
    """
    Search gallery images using model-specific predictions
    Uses TF-IDF on the attributes predicted by the selected model
    
    Args:
        query_text: Text description (e.g., "red leather bag")
        model_type: Model to use (determines which predictions to search)
        top_k: Number of results
    
    Returns:
        List of matching images and info HTML
    """
    global gallery_images, gallery_descriptions, gallery_features, tfidf_vectorizer, tfidf_matrix
    
    if not query_text or query_text.strip() == "":
        return [], "<p style='color: red;'>⚠️ Please enter a search query.</p>"
    
    if len(gallery_images) == 0:
        return [], "<p style='color: red;'>⚠️ No images in gallery. Please upload images first in the Gallery tab.</p>"
    
    # Check if this model's predictions are available
    if model_type not in tfidf_vectorizer or model_type not in tfidf_matrix:
        return [], f"<p style='color: red;'>⚠️ Gallery not indexed for {model_type}. Please reload the UI.</p>"
    
    if model_type not in gallery_descriptions:
        return [], f"<p style='color: red;'>⚠️ No predictions for {model_type}.</p>"
    
    # Get model-specific data
    vectorizer = tfidf_vectorizer[model_type]
    matrix = tfidf_matrix[model_type]
    descriptions = gallery_descriptions[model_type]
    
    # Transform query using THIS model's TF-IDF
    query_vector = vectorizer.transform([query_text])
    
    # Compute cosine similarity between query and THIS model's predictions
    text_similarities = cosine_similarity(query_vector, matrix)[0]
    
    # Rank by similarity
    top_k = min(top_k, len(gallery_images))
    top_indices = np.argsort(text_similarities)[::-1][:top_k]
    scores = [float(text_similarities[i]) for i in top_indices]
    
    # Get matching images and their descriptions (from THIS model)
    result_images = [gallery_images[i] for i in top_indices]
    matched_descriptions = [descriptions[i] for i in top_indices]
    
    # Create info HTML
    model_display = "DeiT-Tiny" if model_type == 'deit_tiny' else "MobileViT-XXS"
    
    info_html = f"""
    <div style="font-family: Arial, sans-serif; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h3 style="margin-top: 0;">🔍 Search Results</h3>
        <p><b>Query:</b> "{query_text}"</p>
        <p><b>Model:</b> {model_display}</p>
        <p><b>Method:</b> TF-IDF on {model_display} predictions</p>
        <p><b>Found:</b> {len(result_images)} matching images (from {len(gallery_images)} total)</p>
        <p style="font-size: 12px; opacity: 0.9; margin-top: 10px;">
            Average similarity: {np.mean(scores):.3f}<br>
            Top match: "{matched_descriptions[0]}" ({scores[0]:.3f})
        </p>
    </div>
    """
    
    return result_images, info_html

# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Image Understanding & Retrieval System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🖼️ Image Understanding & Retrieval System
            ### Deep Learning for Visual Recognition & Hybrid Text-Visual Search
            
            **Features:**
            - **Classification**: Get class & attributes for any image using trained models
            - **Hybrid Retrieval**: Search images using text prompts with both text matching and visual features
            - **Dual Models**: Compare results from DeiT-Tiny & MobileViT-XXS (both 81% accuracy)
            
            **Models**: DeiT-Tiny & MobileViT-XXS
            
            💡 **Gallery auto-loaded** with 600 images analyzed by BOTH models on startup!
            """
        )
        
        with gr.Tabs():
            # Tab 1: Image Classification
            with gr.TabItem("📸 Image Classification"):
                gr.Markdown("### Upload a single image to predict its class and visual attributes")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Upload Image")
                        model_select_classify = gr.Radio(
                            choices=['deit_tiny', 'mobilevit_xxs'],
                            value='mobilevit_xxs',
                            label="Select Model",
                            info="Choose which model to use for prediction"
                        )
                        classify_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        classify_output = gr.HTML(label="Prediction Results")
                
                classify_btn.click(
                    fn=predict_image,
                    inputs=[input_image, model_select_classify],
                    outputs=classify_output
                )
            
            # Tab 2: Text-based Search
            with gr.TabItem("🔎 Text-based Image Search"):
                gr.Markdown(
                    """
                    ### Search Your Images with Text Prompts
                    Search using natural language based on model predictions!
                    
                    **How it works:**
                    - Each model predicts attributes (color, material, class, condition) for all images
                    - Your search uses predictions from YOUR SELECTED model
                    - TF-IDF matches your text query against those predictions
                    
                    **Gallery auto-loaded** with predictions from BOTH models on startup!
                    """
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., 'blue plastic bottle', 'black leather bag', 'red cap'...",
                        lines=2
                    )
                
                with gr.Row():
                    model_select_search = gr.Radio(
                        choices=['deit_tiny', 'mobilevit_xxs'],
                        value='mobilevit_xxs',
                        label="Select Model for Retrieval",
                        info="Choose which model's features to use for search (both are pre-loaded)"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        label="Number of Results",
                        info="How many images to retrieve"
                    )
                
                search_btn = gr.Button("🔍 Search Images", variant="primary", size="lg")
                
                search_info = gr.HTML(label="Search Info")
                search_results = gr.Gallery(
                    label="Search Results",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
                
                search_btn.click(
                    fn=search_images,
                    inputs=[query_input, model_select_search, top_k_slider],
                    outputs=[search_results, search_info]
                )
                
                gr.Markdown(
                    """
                    **💡 Example Queries:**
                    - "blue plastic bottle" - Find blue plastic items
                    - "black leather bag" - Find black leather bags
                    - "red fabric cap" - Find red fabric caps
                    - "worn sneakers" - Find used footwear
                    - "new backpack" - Find new travel items
                    
                    **How it works:**
                    1. Each model makes its OWN predictions for all 600 images
                    2. When you search, it uses predictions from YOUR SELECTED model
                    3. TF-IDF matches your query against that model's predictions
                    4. Different models = Different predictions = Different results!
                    
                    **Try switching models** to see how DeiT-Tiny vs MobileViT-XXS predict differently!
                    """
                )
        
        with gr.Accordion("📊 Supported Categories", open=False):
            gr.Markdown(
                """
                ### 10 Object Classes:
                1. Clothing Cap
                2. Clothing Wrist Watch
                3. Food Storage Plastic Container
                4. Footwear Sneakers
                5. Personal Care Deodorant
                6. Personal Care Shampoo Bottle
                7. Personal Care Soap Bar
                8. Tableware Water Bottle
                9. Travel Backpack
                10. Travel Handbag
                
                ### Colors (15):
                black, blue, brown, gold, gray, green, grey, multicolor, pink, purple, red, silver, transparent, white, yellow
                
                ### Materials (9):
                cotton, fabric, glass, leather, metal, plastic, rubber, soap, synthetic
                
                ### Conditions (4):
                mint, new, used, worn
                """
            )
        
        with gr.Accordion("ℹ️ About the Models", open=False):
            gr.Markdown(
                """
                ## About This System
                
                ### 🧠 Models
                
                **DeiT-Tiny (Data-efficient Image Transformer)**
                - Parameters: 5.7M
                - Accuracy: 81.13%
                - Architecture: Vision Transformer
                - Training: Two-phase fine-tuning on ImageNet
                
                **MobileViT-XXS (Mobile Vision Transformer)**
                - Parameters: 1.2M (5x lighter!)
                - Accuracy: 81.13%
                - Architecture: Lightweight Hybrid CNN-ViT
                - Training: Balanced fine-tuning
                - **Recommended**: Faster and lighter
                
                ### 📊 Tasks
                
                1. **Object Classification**: 10 product categories
                2. **Color Detection**: 15 color options
                3. **Material Recognition**: 9 material types
                4. **Condition Assessment**: 4 condition states
                
                ### 🔧 Technical Details
                
                - **Framework**: PyTorch + timm + Gradio
                - **Training Dataset**: 600 images (10 classes)
                - **Multi-task Learning**: Joint training for all attributes
                - **Test Performance**: 81% accuracy, 0.80 F1 score
                
                ### 💡 How to Use
                
                1. Upload any image (JPG, PNG, etc.)
                2. Select a model (MobileViT-XXS recommended)
                3. Click "Classify Image"
                4. View predictions with confidence scores
                
                ### ⚙️ Requirements
                
                - Only needs model files (.pth)
                - Optional: Place images in `images/` folder for auto-load
                - Works on CPU or GPU
                - Standalone and portable
                """
            )
        
        gr.Markdown(
            """
            ---
            <center>
            <p><b>Image Understanding & Retrieval System (TF-IDF)</b></p>
            <p>Built with PyTorch, timm, scikit-learn, and Gradio</p>
            <p>Fast text-based retrieval without transformers!</p>
            </center>
            """
        )
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMAGE UNDERSTANDING & RETRIEVAL SYSTEM (TF-IDF)")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Models loaded: {list(models.keys())}")
    print(f"\nFeatures:")
    print("  [OK] Image classification (10 classes)")
    print("  [OK] Attribute prediction (color, material, condition)")
    print("  [OK] Auto-load gallery from images folder")
    print("  [OK] Text-based image search (TF-IDF)")
    print("  [OK] No transformers or external datasets needed!")
    
    # Auto-load images from folder on startup with BOTH models
    print("\n" + "="*70)
    print("AUTO-LOADING GALLERY FROM 'images/' FOLDER")
    print("="*70)
    
    # Process with both models
    total_loaded = 0
    for model_name in ['mobilevit_xxs', 'deit_tiny']:
        if models.get(model_name) is not None:
            print(f"\n--- Processing with {model_name} ---")
            num_loaded = load_images_from_folder(folder_path='images', model_type=model_name, show_progress=True)
            if num_loaded > 0 and total_loaded == 0:
                total_loaded = num_loaded  # Count images only once
    
    if total_loaded > 0:
        print(f"\n[SUCCESS] {total_loaded} images analyzed with BOTH models and ready for search!")
    else:
        print("[INFO] No images auto-loaded. You can upload images via the UI.")
    
    print("\n" + "="*70)
    print("Starting Gradio UI...")
    print("="*70 + "\n")
    
    demo = create_ui()
    demo.launch(
        share=False,
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        show_error=True,
        inbrowser=True  # Automatically open browser
    )

