import os

def count_images_in_subfolders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    sum = 0
    for subfolder in subfolders:
        image_count = count_images_in_folder(subfolder)
        sum += image_count
        print(f"Subfolder: {os.path.basename(subfolder)}, Number of Images: {image_count}")
    print(f"Total number of images: {sum}")

def count_images_in_folder(folder):
    image_count = sum(1 for entry in os.scandir(folder) if entry.is_file())
    return image_count

if __name__ == "__main__":
    root_folder = "./garbage"  # Replace with the path to your folder
    count_images_in_subfolders(root_folder)
