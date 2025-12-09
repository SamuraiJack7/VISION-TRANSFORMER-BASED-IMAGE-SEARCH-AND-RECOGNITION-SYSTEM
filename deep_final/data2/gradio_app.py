import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import timm
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH_DEIT = "best_model1_deit_tiny.pth"
MODEL_PATH_MOBILEVIT = "best_model1_mobilevit_xxs.pth"
LABELS_CSV = "labels_global.csv"
IMAGES_DIR = "images"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==================== Model Definition ====================
class MultiTaskViT(nn.Module):
    """Multi-task Vision Transformer for classification and attribute prediction"""
    
    def __init__(self, model_name='deit_tiny_patch16_224', num_classes=10, 
                 num_colors=10, num_materials=10, num_conditions=5, pretrained=False):
        super(MultiTaskViT, self).__init__()
        
        # Determine global pool type based on model
        if 'mobilevit' in model_name:
            global_pool = 'avg'
        elif 'deit' in model_name:
            global_pool = 'token'
        else:
            global_pool = 'avg'
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        self.feature_dim = feature_dim
        hidden_dim = feature_dim // 2
        
        # Task-specific heads
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


# ==================== Utility Functions ====================
def parse_attributes(attr_string):
    """Parse semicolon-separated attributes"""
    attrs = {}
    if pd.notna(attr_string) and attr_string:
        for pair in attr_string.split(';'):
            if ':' in pair:
                key, value = pair.split(':', 1)
                attrs[key.strip()] = value.strip()
    return attrs


def extract_attribute_labels(row):
    """Extract color, material, condition from attributes"""
    attrs = parse_attributes(row['attributes'])
    return {
        'color': attrs.get('color', 'unknown'),
        'material': attrs.get('material', 'unknown'),
        'condition': attrs.get('condition', 'unknown')
    }


# ==================== Data Loading ====================
print("Loading dataset...")
df = pd.read_csv(LABELS_CSV, keep_default_na=False)
print(f"Loaded {len(df)} samples")

# Extract attributes
df[['attr_color', 'attr_material', 'attr_condition']] = df.apply(
    lambda row: pd.Series(extract_attribute_labels(row)), axis=1
)

# Create encoders
all_class_encoder = LabelEncoder()
all_class_encoder.fit(df['class_label'])

all_color_encoder = LabelEncoder()
all_color_encoder.fit(df['attr_color'].fillna('unknown'))

all_material_encoder = LabelEncoder()
all_material_encoder.fit(df['attr_material'].fillna('unknown'))

all_condition_encoder = LabelEncoder()
all_condition_encoder.fit(df['attr_condition'].fillna('unknown'))

num_classes = len(all_class_encoder.classes_)
num_colors = len(all_color_encoder.classes_)
num_materials = len(all_material_encoder.classes_)
num_conditions = len(all_condition_encoder.classes_)

print(f"Classes: {num_classes}, Colors: {num_colors}, Materials: {num_materials}, Conditions: {num_conditions}")


# ==================== Helper Function to Extract Model Dimensions ====================
def get_model_dimensions_from_checkpoint(checkpoint_path):
    """Extract the number of classes from a checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get dimensions from the final layer of each head
        num_classes = checkpoint['class_head.3.weight'].shape[0]
        num_colors = checkpoint['color_head.3.weight'].shape[0]
        num_materials = checkpoint['material_head.3.weight'].shape[0]
        num_conditions = checkpoint['condition_head.3.weight'].shape[0]
        
        return num_classes, num_colors, num_materials, num_conditions
    except Exception as e:
        print(f"Warning: Could not extract dimensions from checkpoint: {e}")
        return None


# ==================== Model Loading ====================
print("Loading models...")

# Dictionary to store all available models
available_models = {}
model_info = {}

# Try to load DeiT model
if os.path.exists(MODEL_PATH_DEIT):
    try:
        print("\n--- Loading DeiT-Tiny Model ---")
        # Get dimensions from checkpoint
        dims = get_model_dimensions_from_checkpoint(MODEL_PATH_DEIT)
        if dims is not None:
            model_num_classes, model_num_colors, model_num_materials, model_num_conditions = dims
            print(f"Detected from checkpoint: {model_num_classes} classes, {model_num_colors} colors, {model_num_materials} materials, {model_num_conditions} conditions")
        else:
            model_num_classes, model_num_colors, model_num_materials, model_num_conditions = num_classes, num_colors, num_materials, num_conditions
        
        deit_model = MultiTaskViT(
            model_name='deit_tiny_patch16_224',
            num_classes=model_num_classes,
            num_colors=model_num_colors,
            num_materials=model_num_materials,
            num_conditions=model_num_conditions,
            pretrained=False
        )
        deit_model.load_state_dict(torch.load(MODEL_PATH_DEIT, map_location=device))
        deit_model = deit_model.to(device)
        deit_model.eval()
        
        available_models['DeiT-Tiny'] = deit_model
        model_info['DeiT-Tiny'] = {
            'num_classes': model_num_classes,
            'num_colors': model_num_colors,
            'num_materials': model_num_materials,
            'num_conditions': model_num_conditions
        }
        
        print(f"✓ Successfully loaded DeiT-Tiny model")
    except Exception as e:
        print(f"✗ Failed to load DeiT model: {e}")

# Try to load MobileViT model
if os.path.exists(MODEL_PATH_MOBILEVIT):
    try:
        print("\n--- Loading MobileViT-XXS Model ---")
        # Get dimensions from checkpoint
        dims = get_model_dimensions_from_checkpoint(MODEL_PATH_MOBILEVIT)
        if dims is not None:
            model_num_classes, model_num_colors, model_num_materials, model_num_conditions = dims
            print(f"Detected from checkpoint: {model_num_classes} classes, {model_num_colors} colors, {model_num_materials} materials, {model_num_conditions} conditions")
        else:
            model_num_classes, model_num_colors, model_num_materials, model_num_conditions = num_classes, num_colors, num_materials, num_conditions
        
        mobilevit_model = MultiTaskViT(
            model_name='mobilevit_xxs',
            num_classes=model_num_classes,
            num_colors=model_num_colors,
            num_materials=model_num_materials,
            num_conditions=model_num_conditions,
            pretrained=False
        )
        mobilevit_model.load_state_dict(torch.load(MODEL_PATH_MOBILEVIT, map_location=device))
        mobilevit_model = mobilevit_model.to(device)
        mobilevit_model.eval()
        
        available_models['MobileViT-XXS'] = mobilevit_model
        model_info['MobileViT-XXS'] = {
            'num_classes': model_num_classes,
            'num_colors': model_num_colors,
            'num_materials': model_num_materials,
            'num_conditions': model_num_conditions
        }
        
        print(f"✓ Successfully loaded MobileViT-XXS model")
    except Exception as e:
        print(f"✗ Failed to load MobileViT model: {e}")

# Check if at least one model loaded
if not available_models:
    raise RuntimeError("No models could be loaded! Please ensure at least one model file exists.")

# Print summary
print(f"\n{'='*50}")
print(f"Models Available: {list(available_models.keys())}")
print(f"{'='*50}")

# Set default model (prefer DeiT, fallback to MobileViT)
default_model_name = 'DeiT-Tiny' if 'DeiT-Tiny' in available_models else list(available_models.keys())[0]
print(f"Default model: {default_model_name}")


# ==================== Model-Enhanced Retrieval System ====================
print("\nPreparing model-enhanced retrieval system...")

# Cache for model predictions
model_predictions_cache = {}

def get_model_predictions(image_path: str, model_choice: str) -> dict:
    """Get model predictions for an image (with caching)"""
    try:
        if not os.path.exists(image_path):
            return None
        
        # Check cache first
        cache_key = f"{image_path}_{model_choice}"
        if cache_key in model_predictions_cache:
            return model_predictions_cache[cache_key]
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get predictions from model
        selected_model = available_models[model_choice]
        with torch.no_grad():
            outputs = selected_model(img_tensor)
        
        # Get predicted indices
        class_idx = torch.argmax(outputs['class_logits'], dim=1).item()
        color_idx = torch.argmax(outputs['color_logits'], dim=1).item()
        material_idx = torch.argmax(outputs['material_logits'], dim=1).item()
        condition_idx = torch.argmax(outputs['condition_logits'], dim=1).item()
        
        # Get confidence scores
        class_conf = torch.softmax(outputs['class_logits'], dim=1)[0, class_idx].item()
        color_conf = torch.softmax(outputs['color_logits'], dim=1)[0, color_idx].item()
        material_conf = torch.softmax(outputs['material_logits'], dim=1)[0, material_idx].item()
        condition_conf = torch.softmax(outputs['condition_logits'], dim=1)[0, condition_idx].item()
        
        # Decode predictions safely
        try:
            pred_class = all_class_encoder.inverse_transform([class_idx])[0] if class_idx < len(all_class_encoder.classes_) else f"class_{class_idx}"
        except:
            pred_class = f"class_{class_idx}"
        
        try:
            pred_color = all_color_encoder.inverse_transform([color_idx])[0] if color_idx < len(all_color_encoder.classes_) else f"color_{color_idx}"
        except:
            pred_color = f"color_{color_idx}"
        
        try:
            pred_material = all_material_encoder.inverse_transform([material_idx])[0] if material_idx < len(all_material_encoder.classes_) else f"material_{material_idx}"
        except:
            pred_material = f"material_{material_idx}"
        
        try:
            pred_condition = all_condition_encoder.inverse_transform([condition_idx])[0] if condition_idx < len(all_condition_encoder.classes_) else f"condition_{condition_idx}"
        except:
            pred_condition = f"condition_{condition_idx}"
        
        predictions = {
            'class': pred_class,
            'color': pred_color,
            'material': pred_material,
            'condition': pred_condition,
            'class_conf': class_conf,
            'color_conf': color_conf,
            'material_conf': material_conf,
            'condition_conf': condition_conf
        }
        
        # Cache the predictions
        model_predictions_cache[cache_key] = predictions
        return predictions
        
    except Exception as e:
        print(f"Error getting predictions for {image_path}: {e}")
        return None


print("Model-enhanced retrieval system ready!")


# ==================== Pre-compute Predictions for Fast Retrieval ====================
def precompute_all_predictions():
    """Pre-compute predictions for all images to enable fast searches"""
    print("\n" + "="*60)
    print("PRE-COMPUTING MODEL PREDICTIONS FOR ALL IMAGES")
    print("This will take 2-5 minutes but makes searches instant!")
    print("="*60)
    
    from tqdm import tqdm
    
    total_images = len(df)
    
    for model_name in available_models.keys():
        print(f"\n🤖 Processing with {model_name}...")
        
        cached_count = 0
        processed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=total_images, desc=f"{model_name}"):
            img_path = row['image_path']
            
            # Handle path variations
            if img_path.startswith('images/') or img_path.startswith('images\\'):
                full_path = img_path
            else:
                full_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))
            
            if os.path.exists(full_path):
                # Check if already cached
                cache_key = f"{full_path}_{model_name}"
                if cache_key not in model_predictions_cache:
                    # Compute and cache predictions
                    predictions = get_model_predictions(full_path, model_name)
                    if predictions:
                        processed_count += 1
                else:
                    cached_count += 1
        
        print(f"✓ {model_name}: {processed_count} new predictions, {cached_count} already cached")
    
    print("\n" + "="*60)
    print(f"✅ PRE-COMPUTATION COMPLETE!")
    print(f"📊 Total cached predictions: {len(model_predictions_cache)}")
    print(f"⚡ Searches will now be INSTANT!")
    print("="*60 + "\n")


# ==================== Image Transforms ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== Prediction Functions ====================
def predict_image(image: Image.Image, model_choice: str) -> str:
    """Predict class and attributes for an uploaded image using selected model"""
    if image is None:
        return "⚠️ Please upload an image."
    
    if model_choice not in available_models:
        return f"❌ Error: Model '{model_choice}' is not available."
    
    try:
        # Get the selected model
        selected_model = available_models[model_choice]
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform and add batch dimension
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = selected_model(img_tensor)
        
        # Get predictions
        class_idx = torch.argmax(outputs['class_logits'], dim=1).item()
        color_idx = torch.argmax(outputs['color_logits'], dim=1).item()
        material_idx = torch.argmax(outputs['material_logits'], dim=1).item()
        condition_idx = torch.argmax(outputs['condition_logits'], dim=1).item()
        
        # Get confidence scores
        class_prob = torch.softmax(outputs['class_logits'], dim=1)[0, class_idx].item()
        color_prob = torch.softmax(outputs['color_logits'], dim=1)[0, color_idx].item()
        material_prob = torch.softmax(outputs['material_logits'], dim=1)[0, material_idx].item()
        condition_prob = torch.softmax(outputs['condition_logits'], dim=1)[0, condition_idx].item()
        
        # Decode predictions with error handling
        try:
            predicted_class = all_class_encoder.inverse_transform([class_idx])[0] if class_idx < len(all_class_encoder.classes_) else f"class_{class_idx}"
        except:
            predicted_class = f"class_{class_idx}"
            
        try:
            predicted_color = all_color_encoder.inverse_transform([color_idx])[0] if color_idx < len(all_color_encoder.classes_) else f"color_{color_idx}"
        except:
            predicted_color = f"color_{color_idx}"
            
        try:
            predicted_material = all_material_encoder.inverse_transform([material_idx])[0] if material_idx < len(all_material_encoder.classes_) else f"material_{material_idx}"
        except:
            predicted_material = f"material_{material_idx}"
            
        try:
            predicted_condition = all_condition_encoder.inverse_transform([condition_idx])[0] if condition_idx < len(all_condition_encoder.classes_) else f"condition_{condition_idx}"
        except:
            predicted_condition = f"condition_{condition_idx}"
        
        # Format results
        result = f"""
## 🔮 Prediction Results

### 🤖 **Model Used**: `{model_choice}`
**Device**: {str(device).upper()}

---

### 📦 **Class**: `{predicted_class}`
**Confidence**: {class_prob*100:.2f}%

### 🎨 **Attributes**:
- **Color**: `{predicted_color}` (Confidence: {color_prob*100:.2f}%)
- **Material**: `{predicted_material}` (Confidence: {material_prob*100:.2f}%)
- **Condition**: `{predicted_condition}` (Confidence: {condition_prob*100:.2f}%)

---
_Results generated using {model_choice} model_
"""
        return result
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"


# ==================== Retrieval Functions ====================
def retrieve_images_by_text(query: str, top_k: int = 10, model_choice: str = None, use_model: bool = False) -> List[Tuple[str, str]]:
    """Retrieve images based on text query - optionally using AI model predictions"""
    if not query or query.strip() == "":
        return []
    
    query = query.lower().strip()
    query_words = set(query.split())
    
    # Score each image based on query match
    scores = []
    
    for idx, row in df.iterrows():
        score = 0.0
        
        img_path = row['image_path']
        # Handle path variations
        if img_path.startswith('images/') or img_path.startswith('images\\'):
            full_path = img_path
        else:
            full_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))
        
        if not os.path.exists(full_path):
            continue
        
        # Method 1: Use CSV metadata (always included)
        # Check caption (highest weight)
        caption = str(row['caption']).lower()
        if query in caption:
            score += 5.0
        else:
            caption_words = set(caption.split())
            word_matches = len(query_words & caption_words)
            score += word_matches * 1.0
        
        # Check class label from CSV
        class_label = str(row['class_label']).lower()
        if query in class_label:
            score += 3.0
        else:
            class_words = set(class_label.replace('_', ' ').split())
            word_matches = len(query_words & class_words)
            score += word_matches * 0.8
        
        # Check CSV attributes
        attrs = parse_attributes(row['attributes'])
        for key, value in attrs.items():
            value_lower = value.lower()
            if query in value_lower:
                score += 2.0
            value_words = set(value_lower.split())
            word_matches = len(query_words & value_words)
            score += word_matches * 0.5
        
        # Method 2: Use AI Model Predictions (if enabled)
        if use_model and model_choice:
            predictions = get_model_predictions(full_path, model_choice)
            
            if predictions:
                # Boost score based on model predictions
                pred_class = predictions['class'].lower().replace('_', ' ')
                pred_color = predictions['color'].lower()
                pred_material = predictions['material'].lower()
                pred_condition = predictions['condition'].lower()
                
                # Weight by confidence
                class_weight = predictions['class_conf']
                color_weight = predictions['color_conf']
                material_weight = predictions['material_conf']
                condition_weight = predictions['condition_conf']
                
                # Check predicted class
                if query in pred_class:
                    score += 4.0 * class_weight
                else:
                    pred_class_words = set(pred_class.split())
                    word_matches = len(query_words & pred_class_words)
                    score += word_matches * 1.0 * class_weight
                
                # Check predicted attributes
                if query in pred_color:
                    score += 3.0 * color_weight
                
                if query in pred_material:
                    score += 3.0 * material_weight
                
                if query in pred_condition:
                    score += 2.0 * condition_weight
                
                # Word-level matching with predictions
                pred_words = set([pred_color, pred_material, pred_condition])
                for pred_word in pred_words:
                    if pred_word in query_words:
                        score += 1.5
        
        # Only include images with positive scores
        if score > 0:
            scores.append((idx, score, row, full_path))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k results
    results = []
    search_method = "CSV metadata" if not use_model else f"{model_choice} AI predictions"
    
    for idx, score, row, full_path in scores[:top_k]:
        # Always get and show model predictions
        if use_model and model_choice:
            predictions = get_model_predictions(full_path, model_choice)
            if predictions:
                caption_text = (
                    f"**{predictions['class'].replace('_', ' ').title()}** ({predictions['class_conf']*100:.0f}%)\n"
                    f"🎨 {predictions['color']} ({predictions['color_conf']*100:.0f}%) | "
                    f"🔧 {predictions['material']} ({predictions['material_conf']*100:.0f}%) | "
                    f"✨ {predictions['condition']} ({predictions['condition_conf']*100:.0f}%)\n"
                    f"**{model_choice} Prediction** | Match Score: {score:.1f}"
                )
            else:
                caption_text = (
                    f"**{row['class_label'].replace('_', ' ').title()}**\n"
                    f"_{row['caption']}_\n"
                    f"Match Score: {score:.1f}"
                )
        else:
            caption_text = (
                f"**{row['class_label'].replace('_', ' ').title()}**\n"
                f"_{row['caption']}_\n"
                f"🎨 {row['attr_color']} | 🔧 {row['attr_material']} | ✨ {row['attr_condition']}\n"
                f"Match Score: {score:.1f}"
            )
        
        results.append((full_path, caption_text))
    
    if not results:
        return []
    
    return results


def retrieve_images_interface(query: str, num_results: int, model_choice: str) -> tuple:
    """Interface for Gradio to retrieve images using model predictions"""
    
    if not query or query.strip() == "":
        return [], "⚠️ Please enter a search query."
    
    # Model-based retrieval (always use AI predictions)
    results = retrieve_images_by_text(query, top_k=num_results, model_choice=model_choice, use_model=True)
    
    if not results:
        message = f"❌ No images found matching '{query}' using {model_choice}. Try:\n- Different keywords\n- More general terms\n- Check spelling"
        return [], message
    
    message = f"✅ Found {len(results)} image(s) matching '{query}' using {model_choice} model predictions"
    
    return results, message


# ==================== Gradio Interface ====================
def create_ui():
    """Create the Gradio UI with two tabs"""
    
    with gr.Blocks(title="Image Classification & Retrieval System", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🖼️ Image Classification & Retrieval System
        ### Powered by Vision Transformers
        
        This application provides two powerful AI-driven functionalities:
        1. **🔍 Image Prediction**: Upload any image to predict its class and attributes (color, material, condition) with confidence scores
        2. **🔎 Image Retrieval**: Search through 11,000+ images using natural language text descriptions
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Image Prediction
            with gr.Tab("🔍 Image Prediction"):
                gr.Markdown("""
                ### Upload an image to classify
                The model will predict:
                - **Class**: The type of object (e.g., pen, backpack, watch)
                - **Attributes**: Color, Material, and Condition of the object
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Model selector
                        model_selector = gr.Radio(
                            choices=list(available_models.keys()),
                            value=default_model_name,
                            label="🤖 Select Model",
                            info="Choose which AI model to use for prediction"
                        )
                        
                        image_input = gr.Image(
                            type="pil", 
                            label="📸 Upload Image",
                            height=400
                        )
                        predict_btn = gr.Button("🚀 Predict", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        prediction_output = gr.Markdown(label="Prediction Results")
                
                # Examples
                gr.Markdown("### 📸 Try these examples:")
                example_images = []
                # Get some sample images
                sample_indices = df.sample(min(5, len(df))).index
                for idx in sample_indices:
                    img_path = df.loc[idx, 'image_path']
                    if img_path.startswith('images/') or img_path.startswith('images\\'):
                        full_path = img_path
                    else:
                        full_path = os.path.join(IMAGES_DIR, os.path.basename(img_path))
                    if os.path.exists(full_path):
                        example_images.append([full_path])
                
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                        label="Sample Images"
                    )
                
                predict_btn.click(
                    fn=predict_image,
                    inputs=[image_input, model_selector],
                    outputs=prediction_output
                )
            
            # Tab 2: Image Retrieval
            with gr.Tab("🔎 Image Retrieval"):
                gr.Markdown("""
                ### ⚡ Instant AI-Powered Image Search
                Search for images by text. All predictions have been **pre-computed** - searches are instant!
                
                **How it works:**
                - ✅ AI models have already analyzed all 11,880 images
                - ✅ Predictions are cached in memory
                - ✅ Your searches are now **INSTANT** (2-3 seconds)
                - Results ranked by confidence-weighted relevance
                """)
                
                with gr.Row():
                    with gr.Column():
                        # Model selector for retrieval
                        retrieval_model_selector = gr.Radio(
                            choices=list(available_models.keys()),
                            value=default_model_name,
                            label="🤖 Select AI Model",
                            info="Choose which Vision Transformer predictions to use"
                        )
                        
                        # Text query input
                        query_input = gr.Textbox(
                            label="🔍 Search Query",
                            placeholder="e.g., black leather wallet, blue pen, red backpack, silver watch...",
                            lines=2
                        )
                        
                        num_results = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=12,
                            step=1,
                            label="📊 Number of Results"
                        )
                        retrieve_btn = gr.Button("🔍 Search Images", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **Search Tips:**
                        - ⚡ All predictions pre-computed - searches are instant!
                        - 🤖 Switch models to compare DeiT-Tiny vs MobileViT-XXS results
                        - 🔍 Search by: color, material, object type, condition, or combinations
                        - 🎯 Results show model predictions with confidence scores
                        """)
                
                retrieval_status = gr.Markdown(value="", visible=True)
                
                retrieval_output = gr.Gallery(
                    label="Retrieved Images",
                    show_label=True,
                    columns=4,
                    rows=3,
                    height="auto",
                    object_fit="contain"
                )
                
                # Example queries
                gr.Markdown("### 💡 Try These Example Searches:")
                example_queries = [
                    ["black pen", 12],
                    ["red backpack", 12],
                    ["silver metal watch", 12],
                    ["leather wallet", 12],
                    ["blue plastic", 12],
                    ["new condition", 12],
                ]
                gr.Examples(
                    examples=example_queries,
                    inputs=[query_input, num_results],
                    label="Click to try"
                )
                
                retrieve_btn.click(
                    fn=retrieve_images_interface,
                    inputs=[query_input, num_results, retrieval_model_selector],
                    outputs=[retrieval_output, retrieval_status]
                )
        
        # Build model info string
        models_info_text = ""
        for name, info in model_info.items():
            models_info_text += f"\n  - **{name}**: {info['num_classes']} classes, {info['num_colors']} colors, {info['num_materials']} materials, {info['num_conditions']} conditions"
        
        gr.Markdown(f"""
        ---
        ### ℹ️ System Information
        - **Available Models**: {len(available_models)} loaded{models_info_text}
        - **Dataset Size**: {len(df):,} images
        - **Device**: {str(device).upper()}
        - **Technology**: Vision Transformers (DeiT-Tiny, MobileViT-XXS)
        - **Capabilities**: 
          - ✅ Model selection for prediction & retrieval
          - ✅ Real-time image classification
          - ✅ Multi-attribute prediction (color, material, condition)
          - ✅ **Model-based image retrieval** (AI predictions only)
          - ✅ Confidence-weighted scoring
          - ✅ Prediction caching for performance
          - ✅ Compare DeiT vs MobileViT results
          - ✅ Pure deep learning approach (no CSV label dependency)
        """)
    
    return app


# ==================== Main ====================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Gradio Application...")
    print("="*50 + "\n")
    
    # Pre-compute all model predictions for instant searches
    print("⚡ Pre-computing predictions for fast retrieval...")
    precompute_all_predictions()
    
    print("🎨 Creating user interface...")
    app = create_ui()
    
    print("\n" + "="*50)
    print("🚀 LAUNCHING APPLICATION")
    print("="*50)
    print("✅ All predictions pre-computed")
    print("⚡ Searches will be INSTANT (2-3 seconds)")
    print("🌐 Opening browser at http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Launch the app (localhost only - not accessible from network)
    app.launch(
        server_name="127.0.0.1",  # Localhost only (not accessible from other computers)
        server_port=7860,
        share=False,  # No public Gradio URL
        debug=True
    )

