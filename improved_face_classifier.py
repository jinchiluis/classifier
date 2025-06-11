import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImprovedFaceClassifier:
    def __init__(self, person1_dir='person1', person2_dir='person2', img_size=(224, 224)):
        """
        Initialize the Improved Face Classifier for small datasets
        
        Args:
            person1_dir: Directory containing person1's face images
            person2_dir: Directory containing person2's face images
            img_size: Input image size
        """
        self.person1_dir = person1_dir
        self.person2_dir = person2_dir
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def prepare_data_advanced(self, batch_size=16, validation_split=0.15):
        """
        Prepare data with aggressive augmentation for small datasets
        
        Args:
            batch_size: Smaller batch size for small datasets
            validation_split: Smaller validation split to keep more training data
        
        Returns:
            train_generator, validation_generator
        """
        parent_dir = os.path.dirname(self.person1_dir)
        if not parent_dir:
            parent_dir = '.'
        
        # Very aggressive augmentation for small datasets
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,  # Increased rotation
            width_shift_range=0.3,  # More shifting
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,  # Don't flip faces vertically
            brightness_range=[0.7, 1.3],  # More brightness variation
            zoom_range=0.3,  # More zoom
            shear_range=0.2,
            channel_shift_range=20,  # Color augmentation
            fill_mode='reflect',  # Better than 'nearest' for faces
            validation_split=validation_split
        )
        
        # Minimal augmentation for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            parent_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            interpolation='bilinear'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            parent_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_indices = train_generator.class_indices
        self.class_names = {v: k for k, v in self.class_indices.items()}
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Classes: {self.class_indices}")
        
        # Warning if dataset is too small
        if train_generator.samples < 100:
            print("\n⚠️  WARNING: Very small dataset detected!")
            print("   Consider collecting more images for better results.")
            print("   Using aggressive augmentation to compensate.\n")
        
        return train_generator, validation_generator
    
    def build_model_for_small_dataset(self, architecture='mobilenet'):
        """
        Build a model optimized for small datasets
        
        Args:
            architecture: 'mobilenet', 'vgg16', or 'resnet50'
        
        Returns:
            Compiled model
        """
        # Choose base model - MobileNet often works better for small datasets
        if architecture == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif architecture == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:  # resnet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        # First, train only the top layers
        base_model.trainable = False
        
        # Build the model with stronger regularization
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.6),  # Higher dropout
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        
        # Compile with higher initial learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.base_model = base_model
        return model
    
    def train_two_stage(self, train_generator, validation_generator, 
                       initial_epochs=20, fine_tune_epochs=30):
        """
        Two-stage training: first train top layers, then fine-tune
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            initial_epochs: Epochs for initial training
            fine_tune_epochs: Epochs for fine-tuning
        
        Returns:
            Training history
        """
        # Stage 1: Train only the top layers
        print("Stage 1: Training top layers only...")
        
        callbacks_stage1 = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model_stage1.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history1 = self.model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=validation_generator,
            callbacks=callbacks_stage1,
            verbose=1
        )
        
        # Stage 2: Fine-tune some layers
        print("\nStage 2: Fine-tuning...")
        
        # Unfreeze some layers
        self.base_model.trainable = True
        
        # For different architectures, unfreeze different amounts
        if isinstance(self.base_model, tf.keras.applications.MobileNetV2):
            # Unfreeze the last 30 layers of MobileNetV2
            for layer in self.base_model.layers[:-30]:
                layer.trainable = False
        elif isinstance(self.base_model, tf.keras.applications.VGG16):
            # Unfreeze the last 4 layers of VGG16
            for layer in self.base_model.layers[:-4]:
                layer.trainable = False
        else:  # ResNet50
            # Unfreeze the last 15 layers
            for layer in self.base_model.layers[:-15]:
                layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_stage2 = [
            EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model_stage2.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        history2 = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks_stage2,
            verbose=1,
            initial_epoch=initial_epochs
        )
        
        # Combine histories
        history = history1
        for key in history.history:
            history.history[key].extend(history2.history[key])
        
        self.history = history
        return history
    
    def plot_training_history(self):
        """Plot training history with stage separation"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add stage separator line if we know where stage 2 started
        if len(self.history.history['accuracy']) > 20:
            ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Fine-tuning start')
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if len(self.history.history['loss']) > 20:
            ax2.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Fine-tuning start')
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=150)
        plt.show()
    
    def evaluate_with_analysis(self, validation_generator):
        """Enhanced evaluation with more insights"""
        predictions = self.model.predict(validation_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = validation_generator.classes
        
        # Calculate per-class accuracy
        from collections import Counter
        true_counter = Counter(y_true)
        correct_counter = Counter()
        
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct_counter[true] += 1
        
        print("\nPer-Class Accuracy:")
        for class_idx, class_name in self.class_names.items():
            total = true_counter[class_idx]
            correct = correct_counter[class_idx]
            accuracy = correct / total if total > 0 else 0
            print(f"{class_name}: {accuracy:.2%} ({correct}/{total})")
        
        # Standard classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=list(self.class_names.values())))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.class_names.values()),
                   yticklabels=list(self.class_names.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Overall metrics
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        # Confidence analysis
        confidences = np.max(predictions, axis=1)
        print(f"Average Confidence: {np.mean(confidences):.2%}")
        print(f"Confidence when Correct: {np.mean(confidences[y_pred == y_true]):.2%}")
        print(f"Confidence when Wrong: {np.mean(confidences[y_pred != y_true]):.2%}")
    
    # Include all the Grad-CAM methods from the original classifier
    def generate_gradcam(self, img_array, class_idx, layer_name=None):
        """Generate Grad-CAM heatmap"""
        if layer_name is None:
            # Auto-detect last conv layer based on architecture
            if isinstance(self.base_model, tf.keras.applications.MobileNetV2):
                layer_name = 'Conv_1'
            elif isinstance(self.base_model, tf.keras.applications.VGG16):
                layer_name = 'block5_conv3'
            else:  # ResNet50
                layer_name = 'conv5_block3_out'
        
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.base_model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        return heatmap
    
    def save_model(self, filepath='improved_face_classifier.h5'):
        """Save the model"""
        if self.model is None:
            print("No model to save.")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        import json
        with open('class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)
        print("Class indices saved")
    
    def load_model(self, filepath='improved_face_classifier.h5'):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        import json
        with open('class_indices.json', 'r') as f:
            self.class_indices = json.load(f)
        self.class_names = {v: k for k, v in self.class_indices.items()}


# Tips for better results with small datasets
def print_tips():
    print("\n" + "="*60)
    print("TIPS FOR BETTER RESULTS WITH SMALL DATASETS:")
    print("="*60)
    print("1. Collect more images (aim for 200+ per person)")
    print("2. Ensure variety in:")
    print("   - Lighting conditions")
    print("   - Facial expressions")
    print("   - Angles (but not extreme)")
    print("   - Backgrounds")
    print("3. Crop images to focus on faces")
    print("4. Remove blurry or low-quality images")
    print("5. Consider using face detection to pre-crop images")
    print("6. Try different architectures (MobileNet often works better)")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Print tips
    print_tips()
    
    # Initialize classifier
    classifier = ImprovedFaceClassifier('person1', 'person2')
    
    # Prepare data with aggressive augmentation
    print("Preparing data with aggressive augmentation...")
    train_gen, val_gen = classifier.prepare_data_advanced(
        batch_size=16,  # Smaller batch size
        validation_split=0.15  # Keep more data for training
    )
    
    # Try different architectures
    print("\nBuilding model (trying MobileNetV2 for small datasets)...")
    model = classifier.build_model_for_small_dataset(architecture='mobilenet')
    print(f"Total parameters: {model.count_params():,}")
    
    # Two-stage training
    print("\nStarting two-stage training...")
    history = classifier.train_two_stage(
        train_gen,
        val_gen,
        initial_epochs=20,
        fine_tune_epochs=30
    )
    
    # Plot results
    print("\nPlotting training history...")
    classifier.plot_training_history()
    
    # Detailed evaluation
    print("\nPerforming detailed evaluation...")
    classifier.evaluate_with_analysis(val_gen)
    
    # Save model
    print("\nSaving model...")
    classifier.save_model('improved_face_classifier.h5')