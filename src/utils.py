# src/utils.py
import pandas as pd
from IPython.display import display, HTML

def generate_framework_report(baseline_history, transfer_history):
    """
    Compiles accuracy and loss dynamics from multiple training runs
    into a clean, presentation-ready comparison matrix.
    """
    print("\nCompiling DeepTrain Framework Performance Report...")
    
    # Extract peak metrics from the baseline run
    base_metrics = {
        "Experiment Model": "Baseline CNN (From Scratch)",
        "Peak Train Acc": f"{max(baseline_history.history['accuracy'])*100:.2f}%",
        "Peak Val Acc": f"{max(baseline_history.history['val_accuracy'])*100:.2f}%",
        "Final Val Loss": f"{baseline_history.history['val_loss'][-1]:.4f}",
        "Architecture Style": "Custom Conv2D + MaxPooling"
    }
    
    # Extract peak metrics from the transfer learning run
    transfer_metrics = {
        "Experiment Model": "MobileNetV2 (Transfer Learning)",
        "Peak Train Acc": f"{max(transfer_history.history['accuracy'])*100:.2f}%",
        "Peak Val Acc": f"{max(transfer_history.history['val_accuracy'])*100:.2f}%",
        "Final Val Loss": f"{transfer_history.history['val_loss'][-1]:.4f}",
        "Architecture Style": "Frozen Base + Custom Dense Head"
    }
    
    # Convert to a DataFrame for structured tabular display
    report_df = pd.DataFrame([base_metrics, transfer_metrics])
    
    # Style it slightly for a clean notebook visualization layout
    styled_table = report_df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'border-color': '#dee2e6',
        'color': '#212529'
    }).hide(axis='index')
    
    return report_df, styled_table