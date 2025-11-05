import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mediapipe as mp
import joblib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Define your paths
train_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_train'
val_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_val'
test_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\augmented_test'

def extract_hand_landmarks(image_path, hands):
    """
    Extract hand landmarks from image using MediaPipe
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract all 21 landmarks (x, y, z coordinates)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + 
                  (point1.y - point2.y)**2 + 
                  (point1.z - point2.z)**2)

def extract_enhanced_features(hand_landmarks):
    """
    Extract additional geometric features
    """
    enhanced_features = []
    
    # Convert list back to landmark points for calculations
    landmarks = []
    for i in range(0, len(hand_landmarks), 3):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        landmarks.append(Point(hand_landmarks[i], hand_landmarks[i+1], hand_landmarks[i+2]))
    
    # Finger lengths (thumb, index, middle, ring, pinky)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        length = calculate_distance(landmarks[tip], landmarks[mcp])
        enhanced_features.append(length)
    
    # Palm size (distance between wrist and middle finger MCP)
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    enhanced_features.append(palm_size)
    
    return enhanced_features

def create_enhanced_dataset(folder_path):
    """
    Create dataset with enhanced features
    """
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    landmarks_list = []
    labels_list = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(full_path))
                
                # Extract basic landmarks
                basic_landmarks = extract_hand_landmarks(full_path, hands)
                
                if basic_landmarks is not None:
                    # Extract enhanced features
                    enhanced_features = extract_enhanced_features(basic_landmarks)
                    
                    # Combine basic landmarks and enhanced features
                    all_features = np.concatenate([basic_landmarks, enhanced_features])
                    
                    landmarks_list.append(all_features)
                    labels_list.append(parent_folder)
    
    hands.close()
    return np.array(landmarks_list), np.array(labels_list)

# MAIN EXECUTION
print("="*50)
print("EXTRACTING HAND LANDMARKS")
print("="*50)

# Extract features
print("Processing training set...")
X_train, y_train = create_enhanced_dataset(train_folder_path)

print("Processing validation set...")
X_val, y_val = create_enhanced_dataset(val_folder_path)

print("Processing test set...")
X_test, y_test = create_enhanced_dataset(test_folder_path)

# Check if we have enough data
if len(X_train) == 0:
    raise ValueError("No hand landmarks were extracted from training data!")

print(f"\nDataset sizes:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
print(f"Number of features per sample: {X_train.shape[1]}")

# Get class information
class_names = sorted(np.unique(y_train))
num_classes = len(class_names)
label_to_idx = {label: idx for idx, label in enumerate(class_names)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")

print("\n" + "="*50)
print("PREPROCESSING DATA")
print("="*50)

# Encode labels to integers
y_train_encoded = np.array([label_to_idx[label] for label in y_train])
y_val_encoded = np.array([label_to_idx[label] for label in y_val])
y_test_encoded = np.array([label_to_idx[label] for label in y_test])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed!")

print("\n" + "="*50)
print("BASIC SVM TRAINING")
print("="*50)

# Train basic SVM classifier
svm_classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)

print("Training SVM...")
svm_classifier.fit(X_train_scaled, y_train_encoded)

# Evaluate on validation set
y_val_pred = svm_classifier.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val_encoded, y_val_pred)

# Evaluate on test set
y_test_pred = svm_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the basic model
basic_model_data = {
    'svm_model': svm_classifier,
    'scaler': scaler,
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'class_names': class_names
}

joblib.dump(basic_model_data, 'basic_enhanced_model.pkl')
print("\nBasic enhanced model saved as 'basic_enhanced_model.pkl'")

print("\n" + "="*50)
print("BASIC MODEL TRAINING COMPLETED!")
print("="*50)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {num_classes}")