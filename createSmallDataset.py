import os
import shutil

# Define source and target folders
source_folder = './trainImages'
target_folder = './trainImagesSmall'

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Get all files in the source folder
all_files = sorted(os.listdir(source_folder))  # sorted for consistency

# Take the first 10000 image files (assuming all are images)
for filename in all_files[:10000]:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(target_folder, filename)
    shutil.copyfile(src_path, dst_path)

print(f"Copied {min(10000, len(all_files))} images to {target_folder}")