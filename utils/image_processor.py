from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

def process_xray_image(image):
    """Preprocess X-ray image for fracture detection"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def enhance_bone_visibility(image):
    """Enhance X-ray image for better bone visibility"""
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('L'))
    else:
        image_np = image
    
    # Apply histogram equalization
    enhanced = cv2.equalizeHist(image_np)
    
    # Apply Gaussian blur for noise reduction
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)

def detect_bone_contours(image):
    """Detect bone contours in X-ray image"""
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('L'))
    else:
        image_np = image
    
    # Apply edge detection
    edges = cv2.Canny(image_np, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def create_heatmap_overlay(original_image, heatmap):
    """Create heatmap overlay on original image"""
    original_np = np.array(original_image)
    heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
    
    # Convert heatmap to jet colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(original_np, 0.7, heatmap_colored, 0.3, 0)
    
    return Image.fromarray(overlay)
