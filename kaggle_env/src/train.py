import os
import joblib
from config import *
from src.model import build_svm_model, build_mobilenet_model

def train_svm(X_train, y_train, model_path):
    print("Training SVM...")
    svm = build_svm_model()
    svm.fit(X_train, y_train)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(svm, model_path)
    print(f"SVM Model saved to {model_path}")
    return svm

def train_mobilenet(train_gen, val_gen, model_path, epochs=5):
    print("Training MobileNetV2...")
    model = build_mobilenet_model(num_classes=2)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    print(f"MobileNet Model saved to {model_path}")
    return model, history
