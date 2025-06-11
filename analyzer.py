import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import classification_report
import tensorflow as tf

class DatasetAnalyzer:
    """Analyze dataset for potential biases and quality issues"""
    
    def __init__(self, person1_dir='person1', person2_dir='person2'):
        self.person1_dir = person1_dir
        self.person2_dir = person2_dir
        self.results = {}
    
    def analyze_dataset(self):
        """Run comprehensive dataset analysis"""
        print("="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        # 1. Count images
        self._count_images()
        
        # 2. Analyze image properties
        self._analyze_image_properties()
        
        # 3. Check for duplicates
        self._check_duplicates()
        
        # 4. Analyze brightness and quality
        self._analyze_image_quality()
        
        # 5. Visualize samples
        self._visualize_samples()
        
        # 6. Generate report
        self._generate_report()
    
    def _count_images(self):
        """Count images in each class"""
        person1_images = [f for f in os.listdir(self.person1_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        person2_images = [f for f in os.listdir(self.person2_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.results['counts'] = {
            'person1': len(person1_images),
            'person2': len(person2_images)
        }
        
        print(f"\n1. IMAGE COUNTS:")
        print(f"   Person1: {len(person1_images)} images")
        print(f"   Person2: {len(person2_images)} images")
        print(f"   Ratio: {len(person1_images)/len(person2_images):.2f}:1")
        
        if abs(len(person1_images) - len(person2_images)) > 10:
            print("   ⚠️  WARNING: Significant class imbalance detected!")
    
    def _analyze_image_properties(self):
        """Analyze image dimensions, formats, etc."""
        print(f"\n2. IMAGE PROPERTIES:")
        
        for person_dir, person_name in [(self.person1_dir, 'Person1'), 
                                        (self.person2_dir, 'Person2')]:
            sizes = []
            formats = defaultdict(int)
            
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        with Image.open(img_path) as img:
                            sizes.append(img.size)
                            formats[img.format] += 1
                    except:
                        print(f"   ⚠️  Corrupt image: {img_path}")
            
            # Analyze sizes
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                print(f"\n   {person_name}:")
                print(f"   - Avg dimensions: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
                print(f"   - Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
                print(f"   - Formats: {dict(formats)}")
                
                self.results[f'{person_name.lower()}_sizes'] = sizes
    
    def _check_duplicates(self):
        """Check for duplicate or very similar images"""
        print(f"\n3. DUPLICATE CHECK:")
        
        for person_dir, person_name in [(self.person1_dir, 'Person1'), 
                                        (self.person2_dir, 'Person2')]:
            hashes = {}
            duplicates = 0
            
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        img_hash = hash(img.tobytes())
                        if img_hash in hashes:
                            duplicates += 1
                        hashes[img_hash] = img_file
                    except:
                        pass
            
            print(f"   {person_name}: {duplicates} potential duplicates")
    
    def _analyze_image_quality(self):
        """Analyze brightness, contrast, and blur"""
        print(f"\n4. IMAGE QUALITY ANALYSIS:")
        
        quality_stats = {}
        
        for person_dir, person_name in [(self.person1_dir, 'Person1'), 
                                        (self.person2_dir, 'Person2')]:
            brightness_scores = []
            blur_scores = []
            
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        # Read image
                        img = cv2.imread(img_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Brightness (mean pixel value)
                        brightness = np.mean(gray)
                        brightness_scores.append(brightness)
                        
                        # Blur detection (variance of Laplacian)
                        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                        blur_scores.append(blur)
                    except:
                        pass
            
            if brightness_scores:
                quality_stats[person_name] = {
                    'brightness': np.mean(brightness_scores),
                    'blur': np.mean(blur_scores)
                }
                
                print(f"\n   {person_name}:")
                print(f"   - Avg brightness: {np.mean(brightness_scores):.1f} (0-255)")
                print(f"   - Avg sharpness: {np.mean(blur_scores):.1f} (higher=sharper)")
                
                if np.mean(brightness_scores) < 50:
                    print(f"   ⚠️  WARNING: Images are very dark!")
                elif np.mean(brightness_scores) > 200:
                    print(f"   ⚠️  WARNING: Images are overexposed!")
                
                if np.mean(blur_scores) < 100:
                    print(f"   ⚠️  WARNING: Images appear blurry!")
        
        self.results['quality'] = quality_stats
    
    def _visualize_samples(self):
        """Visualize random samples from each class"""
        print(f"\n5. VISUALIZING SAMPLES...")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Random Samples from Each Class', fontsize=16)
        
        for row, (person_dir, person_name) in enumerate([(self.person1_dir, 'Person1'), 
                                                         (self.person2_dir, 'Person2')]):
            # Get random images
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sample_images = np.random.choice(images, min(5, len(images)), replace=False)
            
            for col, img_file in enumerate(sample_images):
                img_path = os.path.join(person_dir, img_file)
                try:
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].axis('off')
                    if col == 0:
                        axes[row, col].set_ylabel(person_name, fontsize=12, rotation=0, labelpad=50)
                except:
                    axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _generate_report(self):
        """Generate summary report with recommendations"""
        print("\n" + "="*60)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        # Check for major issues
        issues = []
        
        # 1. Class imbalance
        count_ratio = self.results['counts']['person1'] / self.results['counts']['person2']
        if count_ratio > 1.5 or count_ratio < 0.67:
            issues.append("SEVERE CLASS IMBALANCE - This will cause bias!")
            issues.append(f"  → Balance datasets or use class weights")
        
        # 2. Quality differences
        if 'quality' in self.results:
            q1 = self.results['quality'].get('Person1', {})
            q2 = self.results['quality'].get('Person2', {})
            
            if q1 and q2:
                brightness_diff = abs(q1['brightness'] - q2['brightness'])
                sharpness_diff = abs(q1['blur'] - q2['blur'])
                
                if brightness_diff > 30:
                    issues.append("SIGNIFICANT BRIGHTNESS DIFFERENCE between classes")
                    issues.append("  → Model might learn lighting instead of faces")
                
                if sharpness_diff > 50:
                    issues.append("SIGNIFICANT SHARPNESS DIFFERENCE between classes")
                    issues.append("  → Model might learn image quality instead of faces")
        
        # 3. Dataset size
        total_images = self.results['counts']['person1'] + self.results['counts']['person2']
        if total_images < 100:
            issues.append("VERY SMALL DATASET")
            issues.append("  → Collect more images or use data augmentation")
        
        if issues:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ No major issues detected!")
        
        print("\n💡 GENERAL RECOMMENDATIONS:")
        print("   1. Ensure both classes have similar:")
        print("      - Number of images (±10%)")
        print("      - Image quality and sharpness")
        print("      - Lighting conditions")
        print("      - Background variety")
        print("   2. Use face detection to crop images")
        print("   3. Remove duplicates and poor quality images")
        print("   4. Aim for 200+ images per person")


def fix_model_bias(model_path='face_classifier_model.h5'):
    """Test the model with non-face images to detect bias"""
    print("\n" + "="*60)
    print("BIAS DETECTION TEST")
    print("="*60)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create test images
    test_images = []
    test_labels = []
    
    # 1. Solid color images
    print("\nTesting with non-face images:")
    for color, name in [([255, 255, 255], 'white'), 
                        ([0, 0, 0], 'black'),
                        ([128, 128, 128], 'gray')]:
        img = np.full((224, 224, 3), color, dtype=np.uint8)
        test_images.append(img)
        test_labels.append(f"Solid {name}")
    
    # 2. Random noise
    noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_images.append(noise)
    test_labels.append("Random noise")
    
    # 3. Simple patterns
    gradient = np.tile(np.linspace(0, 255, 224), (224, 1))
    gradient = np.stack([gradient]*3, axis=-1).astype(np.uint8)
    test_images.append(gradient)
    test_labels.append("Gradient")
    
    # Make predictions
    import json
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    
    print("\nPredictions on non-face images:")
    print("-" * 40)
    
    for img, label in zip(test_images, test_labels):
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        pred_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        print(f"{label:15} → {pred_class} ({confidence:.1%})")
    
    print("-" * 40)
    print("\n⚠️  If all non-face images predict the same person with")
    print("   high confidence, the model has learned biases instead")
    print("   of facial features!")


if __name__ == "__main__":
    # Analyze dataset
    analyzer = DatasetAnalyzer('person1', 'person2')
    analyzer.analyze_dataset()
    
    # Test for bias
    if os.path.exists('face_classifier_model.h5'):
        fix_model_bias('face_classifier_model.h5')
    
    print("\n" + "="*60)
    print("SOLUTIONS FOR BIAS:")
    print("="*60)
    print("1. **Balance your dataset**:")
    print("   - Ensure equal numbers of images")
    print("   - Match image quality between classes")
    print("")
    print("2. **Preprocess consistently**:")
    print("   - Use face detection to crop faces only")
    print("   - Normalize brightness/contrast")
    print("")
    print("3. **Use class weights during training**:")
    print("   class_weight = {0: 1.0, 1: person1_count/person2_count}")
    print("")
    print("4. **Add regularization**:")
    print("   - Increase dropout")
    print("   - Add L2 regularization")
    print("   - Use data augmentation")
    print("="*60)