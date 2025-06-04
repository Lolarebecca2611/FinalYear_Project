import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from utils import (
    preprocess_image,
    extract_minutiae,
    match_fingerprints,
    visualize_features
)

class FingerprintModel:
    def __init__(self, fingerprints_dir: str = "fingerprints"):
        """
        Initialize the fingerprint recognition model.
        
        Args:
            fingerprints_dir: Directory containing fingerprint images
        """
        self.fingerprints_dir = fingerprints_dir
        self.database: Dict[str, List[Dict[str, float]]] = {}
        
        # Create fingerprints directory if it doesn't exist
        os.makedirs(fingerprints_dir, exist_ok=True)
    
    def add_fingerprint(self, image_path: str, person_id: str) -> bool:
        """
        Add a new fingerprint to the database.
        
        Args:
            image_path: Path to the fingerprint image
            person_id: Unique identifier for the person
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                return False
            
            # Process the image and extract features
            processed_image = preprocess_image(image)
            minutiae = extract_minutiae(processed_image)
            
            if not minutiae:
                print(f"Error: No features extracted from {image_path}")
                return False
            
            # Store in database
            self.database[person_id] = minutiae
            
            # Save processed image for visualization
            vis_image = visualize_features(image, minutiae)
            output_path = os.path.join(
                self.fingerprints_dir,
                f"{person_id}_processed.jpg"
            )
            cv2.imwrite(output_path, vis_image)
            
            return True
            
        except Exception as e:
            print(f"Error adding fingerprint: {str(e)}")
            return False
    
    def identify_fingerprint(self, image_path: str, 
                           threshold: float = 0.8) -> Tuple[str, float]:
        """
        Identify a fingerprint by matching against the database.
        
        Args:
            image_path: Path to the fingerprint image to identify
            threshold: Matching threshold
            
        Returns:
            Tuple of (person_id, confidence_score)
        """
        try:
            # Read and process the input image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
            
            processed_image = preprocess_image(image)
            minutiae = extract_minutiae(processed_image)
            
            if not minutiae:
                raise ValueError("No features extracted from input image")
            
            # Match against all fingerprints in database
            best_match = None
            best_confidence = 0.0
            
            for person_id, stored_minutiae in self.database.items():
                is_match, confidence = match_fingerprints(
                    minutiae, stored_minutiae, threshold
                )
                
                if is_match and confidence > best_confidence:
                    best_match = person_id
                    best_confidence = confidence
            
            return best_match, best_confidence
            
        except Exception as e:
            print(f"Error identifying fingerprint: {str(e)}")
            return None, 0.0
    
    def visualize_matching(self, image_path: str, person_id: str = None):
        """
        Visualize the fingerprint matching process.
        
        Args:
            image_path: Path to the fingerprint image
            person_id: Optional person ID to compare against
        """
        try:
            # Read and process the input image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
            
            processed_image = preprocess_image(image)
            minutiae = extract_minutiae(processed_image)
            
            # Create visualization
            vis_image = visualize_features(image, minutiae)
            
            # If person_id is provided, show comparison
            if person_id and person_id in self.database:
                stored_minutiae = self.database[person_id]
                is_match, confidence = match_fingerprints(
                    minutiae, stored_minutiae
                )
                
                # Add text to visualization
                text = f"Match: {is_match}, Confidence: {confidence:.2f}"
                cv2.putText(
                    vis_image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            
            # Display the visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing matching: {str(e)}")

def main():
    # Example usage
    model = FingerprintModel()
    
    # Add some example fingerprints to the database
    # Note: You'll need to provide actual fingerprint images
    print("Please add fingerprint images to the 'fingerprints' directory")
    print("Then use the model as follows:")
    print("\n1. Add a fingerprint to the database:")
    print("   model.add_fingerprint('fingerprints/person1.jpg', 'person1')")
    print("\n2. Identify a fingerprint:")
    print("   person_id, confidence = model.identify_fingerprint('fingerprints/unknown.jpg')")
    print("\n3. Visualize matching:")
    print("   model.visualize_matching('fingerprints/unknown.jpg', 'person1')")

if __name__ == "__main__":
    main() 