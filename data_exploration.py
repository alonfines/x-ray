import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# --- Configuration Paths ---
TRAIN_CSV_PATH = '/gpfs0/tamyr/projects/data/chest_xray/chexpert/train.csv'
VALID_CSV_PATH = '/gpfs0/tamyr/projects/data/chest_xray/chexpert/valid.csv'
BASE_IMAGE_DIR = '/gpfs0/tamyr/projects/data/chest_xray/chexpert/'

# Output directory for saving plots
OUTPUT_DIR = '/gpfs0/tamyr/users/alonfi/XRay'

# Load dataframes
print("Loading dataframes...")
train_df = pd.read_csv(TRAIN_CSV_PATH)
valid_df = pd.read_csv(VALID_CSV_PATH)

# Standard CheXpert label columns
PATHOLOGY_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices'
]

def get_absolute_image_path(csv_path):
    """
    Converts a path like 'CheXpert-v1.0/train/patient...jpg'
    into '/gpfs0/tamyr/projects/data/chest_xray/chexpert/train/patient...jpg'
    """
    # Split on the first '/' to remove the 'CheXpert-v1.0' prefix
    parts = csv_path.split('/', 1)
    if len(parts) > 1:
        # Join the base path with the remaining path
        return os.path.join(BASE_IMAGE_DIR, parts[1])
    return os.path.join(BASE_IMAGE_DIR, csv_path)

def plot_image_and_labels(dataset_type, index):
    """
    Plots the image and its pathology labels, saving it to the OUTPUT_DIR.
    """
    if dataset_type == 'train':
        df = train_df
    elif dataset_type == 'valid':
        df = valid_df
    else:
        raise ValueError("dataset_type must be either 'train' or 'valid'.")
        
    if index >= len(df) or index < 0:
        raise IndexError(f"Index {index} is out of bounds. {dataset_type} has {len(df)} rows.")
        
    row = df.iloc[index]
    
    csv_img_path = row['Path']
    actual_img_path = get_absolute_image_path(csv_img_path)
    
    labels = row[PATHOLOGY_COLUMNS].to_dict()
    
    label_text = "Pathology Labels:\n" + "-"*20 + "\n"
    for k, v in labels.items():
        if pd.notna(v):
            label_text += f"{k}: {v}\n"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        img = mpimg.imread(actual_img_path)
        ax.imshow(img, cmap='gray')
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"IMAGE NOT FOUND AT:\n{actual_img_path}", 
                ha='center', va='center', wrap=True, color='red')
        
    ax.axis('off')
    
    view_type = row.get('Frontal/Lateral', 'Unknown')
    sex = row.get('Sex', 'Unknown')
    age = row.get('Age', 'Unknown')
    
    ax.set_title(f"Dataset: {dataset_type.upper()} | Index: {index}\n"
                 f"Demographics: {sex}, Age {age} | View: {view_type}", 
                 loc='left', fontweight='bold')
    
    ax.text(1.05, 0.5, label_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.7))
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the plot
    output_filename = os.path.join(OUTPUT_DIR, f"{dataset_type}_image_{index}.png")
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Successfully saved image plot to {output_filename}")
    
    # Close the plot to free memory
    plt.close(fig)

def plot_label_histogram(dataset_type):
    """
    Plots a histogram of the positive (1.0) pathology labels for the specified dataset.
    """
    if dataset_type == 'train':
        df = train_df
    elif dataset_type == 'valid':
        df = valid_df
    else:
        raise ValueError("dataset_type must be either 'train' or 'valid'.")

    # Count only the positive (1.0) findings for each pathology
    positive_counts = (df[PATHOLOGY_COLUMNS] == 1.0).sum()

    # Sort them for better visualization
    positive_counts = positive_counts.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    positive_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title(f'Frequency of Positive Pathology Labels in {dataset_type.upper()} Dataset', fontweight='bold')
    plt.xlabel('Pathology')
    plt.ylabel('Total Count (Positive = 1.0)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_filename = os.path.join(OUTPUT_DIR, f"{dataset_type}_label_histogram.png")
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Successfully saved histogram to {output_filename}")
    
    plt.close()

if __name__ == "__main__":
    # Generate and save the 1st image from the train and valid datasets
    plot_image_and_labels('train', 0)
    plot_image_and_labels('valid', 0)
    
    # Generate and save the label histograms
    plot_label_histogram('train')
    plot_label_histogram('valid')