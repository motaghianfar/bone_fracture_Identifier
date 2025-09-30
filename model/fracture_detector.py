import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import os

class FractureDetector:
    def __init__(self, num_classes=1):
        self.model = None
        self.num_classes = num_classes
        self.weights_path = "weights/fracture_model.pth"
        
    def create_model(self):
        """Create ResNet-50 model for fracture detection"""
        try:
            # New torchvision weights system
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base_model = models.resnet50(weights=weights)
        except:
            # Legacy system
            base_model = models.resnet50(pretrained=True)
        
        # Modify final layer for binary classification (fracture vs no fracture)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()
        )
        
        return base_model
    
    def load_model(self, weights_path=None):
        """Load model with weights"""
        if weights_path:
            self.weights_path = weights_path
            
        self.model = self.create_model()
        
        # Load trained weights if available
        if os.path.exists(self.weights_path):
            try:
                checkpoint = torch.load(self.weights_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print("✅ Fracture detection weights loaded successfully!")
            except Exception as e:
                print(f"⚠️ Could not load custom weights: {e}")
                print("✅ Using ImageNet pre-trained weights")
        else:
            print("✅ Using ImageNet pre-trained ResNet-50 weights")
        
        self.model.eval()
        return self.model
    
    def predict(self, image_tensor):
        """Make fracture prediction"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            fracture_prob = outputs.cpu().numpy()[0][0]
            return fracture_prob
    
    def predict_with_heatmap(self, image_tensor):
        """Make prediction and generate simple heatmap"""
        fracture_prob = self.predict(image_tensor)
        
        # Generate a simple heatmap (in real implementation, use Grad-CAM)
        # This is a placeholder - you'd implement proper Grad-CAM here
        heatmap = np.random.rand(224, 224) * fracture_prob
        
        return fracture_prob, heatmap
