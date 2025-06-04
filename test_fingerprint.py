import cv2
import numpy as np
from fingerprint_model import FingerprintModel

def generate_synthetic_fingerprint(size=(300, 300), noise_level=0.1, seed=None, pattern_type=None):
    """
    Generate a synthetic fingerprint image for testing.
    
    Args:
        size: Size of the image (width, height)
        noise_level: Amount of noise to add
        seed: Random seed for reproducibility
        pattern_type: Type of ridge pattern to generate
        
    Returns:
        Synthetic fingerprint image
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create base image with random noise
    image = np.random.normal(128, 30, size).astype(np.uint8)
    
    # Add some ridge patterns
    x = np.linspace(0, 2*np.pi, size[0])
    y = np.linspace(0, 2*np.pi, size[1])
    X, Y = np.meshgrid(x, y)
    
    # Create different ridge patterns based on pattern_type
    if pattern_type is None:
        pattern_type = np.random.choice(['whorl', 'loop', 'arch'])
    
    if pattern_type == 'whorl':
        # Circular pattern
        ridges = np.sin(np.sqrt(X**2 + Y**2) * 2) * 50
    elif pattern_type == 'loop':
        # Curved pattern
        phase = np.random.uniform(0, 2*np.pi)
        ridges = np.sin(X*2 + phase) * np.cos(Y*2) * 50
    else:  # arch
        # Straight pattern
        ridges = np.sin(X*2) * 50
    
    image = cv2.add(image, ridges.astype(np.uint8))
    
    # Add some noise
    noise = np.random.normal(0, noise_level * 255, size).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Apply some blur to make it look more realistic
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize to 0-255 range
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image

def test_fingerprint_system():
    """Test the fingerprint recognition system with synthetic images."""
    # Initialize the model
    model = FingerprintModel()
    
    # Generate and save synthetic fingerprints
    print("Generating synthetic fingerprints...")
    
    # Generate 3 different fingerprints with their modified versions
    pattern_types = ['whorl', 'loop', 'arch']
    for i in range(3):
        # Use the same seed and pattern type for base and modified versions
        seed = i * 100
        pattern = pattern_types[i]
        
        # Generate base fingerprint
        fingerprint = generate_synthetic_fingerprint(
            seed=seed, 
            pattern_type=pattern
        )
        cv2.imwrite(f'fingerprints/person{i+1}.jpg', fingerprint)
        
        # Generate a slightly modified version with the same seed and pattern
        modified = generate_synthetic_fingerprint(
            seed=seed, 
            pattern_type=pattern,
            noise_level=0.15
        )
        cv2.imwrite(f'fingerprints/person{i+1}_modified.jpg', modified)
        
        print(f"Generated {pattern} pattern fingerprint {i+1} and its modified version")
    
    # Add fingerprints to database
    print("\nAdding fingerprints to database...")
    for i in range(3):
        success = model.add_fingerprint(f'fingerprints/person{i+1}.jpg', f'person{i+1}')
        print(f"Added person{i+1} to database: {'Success' if success else 'Failed'}")
    
    # Test identification
    print("\nTesting fingerprint identification...")
    for i in range(3):
        # Try to identify the modified version
        person_id, confidence = model.identify_fingerprint(
            f'fingerprints/person{i+1}_modified.jpg',
            threshold=0.5  # Lower threshold for synthetic fingerprints
        )
        print(f"\nTesting modified fingerprint {i+1}:")
        print(f"Identified as: {person_id}")
        print(f"Confidence: {confidence:.2f}")
        
        # Visualize the matching
        print(f"Visualizing match for person{i+1}...")
        model.visualize_matching(
            f'fingerprints/person{i+1}_modified.jpg',
            f'person{i+1}'
        )

if __name__ == "__main__":
    test_fingerprint_system() 