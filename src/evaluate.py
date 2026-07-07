# src/evaluate.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_saved_model(model_path, test_ds, class_names):
    """
    Loads a saved .keras model, runs inference on the test dataset,
    and returns metrics along with the confusion matrix data.
    """
    print(f"Loading best saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("Running inference on test dataset...")
    y_true = []
    y_pred = []
    
    # Iterate through the test dataset to collect true labels and predictions
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        
        # Convert one-hot encoded arrays back to single class integer IDs
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Generate structural reports
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm, y_true, y_pred

def plot_confusion_matrix(cm, class_names):
    """
    Plots a highly scannable, clean confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix: DeepTrain Baseline Model')
    plt.ylabel('True Class Category')
    plt.xlabel('Predicted Class Category')
    plt.tight_layout()
    plt.show()