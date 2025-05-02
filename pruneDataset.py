import os
import numpy as np
from PIL import Image

# Configuration
folder_path = "trainImages"
tolerance = 1e-3

# Track removed files
removed_files = []

# Process files
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Load image and convert to grayscale for simplicity
            image = Image.open(file_path).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            min_val = image_array.min()
            max_val = image_array.max()

            if abs(max_val - min_val) <= tolerance:
                os.remove(file_path)
                removed_files.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Print results
if removed_files:
    print("Removed files due to low variance (min ≈ max):")
    for f in removed_files:
        print(f)
else:
    print("No files were removed.")