import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

# ========= CONFIG =========
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
DATASET_DIR = os.path.join(BASE_DIR, "data", "bubbles")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bubble_model.pkl")
IMAGE_SIZE = (64, 64)

# ========= FEATURE EXTRACTOR =========
def extract_features(img, image_size=(64,64)):
    img_resized = resize(img, image_size)
    features, _ = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

# ========= LOAD DATA =========
def load_images(split="train"):
    X, y = [], []
    classes = os.listdir(DATASET_DIR)

    for label in classes:
        class_dir = os.path.join(DATASET_DIR, label, split)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = imread(img_path, as_gray=True)
                features = extract_features(img, IMAGE_SIZE)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
    
    return np.array(X), np.array(y)

# ========= MAIN =========
# Load training data
X, y = load_images("train")
print(f"✅ Loaded {len(X)} training images across {len(set(y))} classes.")

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("🚀 Training model with HOG features...")
clf = SVC(kernel="linear", probability=True, class_weight="balanced")
clf.fit(X_train, y_train)

# Evaluate on validation set
y_pred = clf.predict(X_val)
print("\n📊 Validation Report:")
print(classification_report(y_val, y_pred))

# Save trained model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# ========= TEST SET EVALUATION =========
X_test, y_test = load_images("test")
if len(X_test) > 0:
    print(f"\n🔎 Loaded {len(X_test)} test images.")
    y_test_pred = clf.predict(X_test)
    print("\n📊 Test Set Report:")
    print(classification_report(y_test, y_test_pred))
else:
    print("\n⚠️ No test set found in data/bubbles/*/test/")
