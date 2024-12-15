import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import os

class ClothingSegmentation:
    def __init__(self, confidence_threshold=0.5, person_threshold=0.7):
        # Initialize model with weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Option 1: Using pre-trained weights
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS  # Better for instance segmentation
        # Alternative options:
        # weights = DeepLabV3_ResNet50_Weights.DEFAULT  # General purpose
        # weights = DeepLabV3_ResNet50_Weights.ADE20K_V1  # Better for scene understanding
        
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.eval().to(self.device)
        
        # Confidence thresholds
        self.confidence_threshold = confidence_threshold
        self.person_threshold = person_threshold
        
        # Get the transform from the weights
        self.transform = weights.transforms()
        
        # Class indices and weights for different parts
        self.class_weights = {
            15: 1.2,  # person (increased weight)
            1: 1.0,   # clothing base weight
        }
    
    def adjust_prediction_weights(self, output):
        """Apply custom weights to model predictions"""
        # Convert outputs to probabilities
        probs = F.softmax(output, dim=0)
        
        # Apply class-specific weights
        weighted_probs = probs.clone()
        for class_idx, weight in self.class_weights.items():
            weighted_probs[class_idx] *= weight
            
        return weighted_probs
    
    def segment_clothing(self, image_path):
        """Perform segmentation with confidence thresholding"""
        # Preprocess image
        input_batch, original_image, original_size = self.preprocess_image(image_path)
        
        # Get model predictions
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Apply custom weights to predictions
        weighted_probs = self.adjust_prediction_weights(output)
        
        # Get predictions with confidence threshold
        predictions = weighted_probs.argmax(0).cpu().numpy()
        confidence_mask = (weighted_probs.max(0)[0].cpu().numpy() > self.confidence_threshold)
        
        # Convert original image to numpy array for size reference
        image_np = np.array(original_image)
        
        # Resize predictions and confidence mask to match original image size
        predictions_resized = cv2.resize(predictions.astype(np.uint8), 
                                       (image_np.shape[1], image_np.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        confidence_mask_resized = cv2.resize(confidence_mask.astype(np.uint8), 
                                           (image_np.shape[1], image_np.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
        
        # Create masks for different regions with confidence thresholding
        masks = {}
        person_mask = (predictions_resized == 15).astype(np.uint8)
        person_mask = person_mask & (confidence_mask_resized > self.person_threshold)
        
        # Apply morphological operations to improve mask quality
        kernel = np.ones((5,5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        
        # Use connected components with area filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(person_mask)
        
        # Filter components based on area
        min_area = 1000  # Minimum area threshold
        for label in range(1, num_labels):  # Skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area > min_area:
                mask = (labels == label).astype(np.uint8) * 255
                masks[f'clothing_item_{label}'] = mask
        
        return masks, original_image

    def preprocess_image(self, image_path):
        """Load and preprocess the image with custom preprocessing"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Apply transforms
        input_tensor = self.transform(image)
        
        # Add custom normalization if needed
        # input_tensor = self.custom_normalize(input_tensor)
        
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        return input_batch, image, original_size
    
    def create_masked_segments(self, image, masks, output_dir="output_segments"):
        """Create and save masked segments using original image colors"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Dictionary to store all masked segments
        masked_segments = {}
        
        # Create masked segment for each detected region
        for item, mask in masks.items():
            if mask.any():  # Only process if mask contains any detected pixels
                # Ensure mask has same size as image
                mask_resized = cv2.resize(mask, 
                                        (image_np.shape[1], image_np.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Create a copy of the original image
                masked_image = np.zeros_like(image_np)
                
                # Create binary mask
                binary_mask = mask_resized > 0
                
                # Apply mask to original image
                masked_image[binary_mask] = image_np[binary_mask]
                
                # Store masked segment
                masked_segments[item] = masked_image
                
                # Save individual masked segment
                output_path = os.path.join(output_dir, f"{item}.png")
                cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        
        # Create a visualization with all segments side by side
        return self.create_segment_visualization(masked_segments, output_dir)

    
    def create_segment_visualization(self, masked_segments, output_dir):
        """Create a visualization showing all segments side by side"""
        if not masked_segments:
            print("No segments were detected!")
            return None
            
        # Get dimensions of first segment to use as reference
        first_segment = list(masked_segments.values())[0]
        height, width = first_segment.shape[:2]
        
        # Calculate layout
        n_segments = len(masked_segments)
        n_cols = min(3, n_segments)  # Maximum 3 columns
        n_rows = (n_segments + n_cols - 1) // n_cols
        
        # Create canvas for visualization
        canvas = np.zeros((height * n_rows, width * n_cols, 3), dtype=np.uint8)
        
        # Place each segment on the canvas
        for idx, (item, segment) in enumerate(masked_segments.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            y_start = row * height
            y_end = (row + 1) * height
            x_start = col * width
            x_end = (col + 1) * width
            
            canvas[y_start:y_end, x_start:x_end] = segment
            
            # Add label
            cv2.putText(canvas, item, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = os.path.join(output_dir, "all_segments.png")
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        return canvas
    
    # ... (rest of the class remains the same)

def main():
    try:
        # Initialize segmentation class with custom thresholds
        print("Initializing segmentation model...")
        
        # Example configurations:
        
        # Configuration 1: High precision (fewer but more accurate segments)
        segmenter = ClothingSegmentation(confidence_threshold=0.5, person_threshold=0.7)
        
        # Configuration 2: High recall (more segments, might include some noise)
        # segmenter = ClothingSegmentation(confidence_threshold=0.3, person_threshold=0.5)
        
        # Configuration 3: Balanced
        # segmenter = ClothingSegmentation(confidence_threshold=0.5, person_threshold=0.7)
        
        # Process image
        image_path = "tst2.jpg"
        print(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        masks, original_image = segmenter.segment_clothing(image_path)
        
        if not masks:
            print("No clothing items were detected in the image.")
            return
        
        visualization = segmenter.create_masked_segments(original_image, masks)
        
        if visualization is not None:
            print("Processing complete! Check the 'output_segments' directory for results.")
        else:
            print("No visualization was created due to lack of detected segments.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
