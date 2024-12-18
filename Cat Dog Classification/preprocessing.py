# Import necessary libraries 
import os
import cv2
import filetype
from concurrent.futures import ThreadPoolExecutor
from config import Config


def unzip_folder(folder_name):
    """
    TODO.
    Currently, i have unzipped the folder myself but i want to automate this process
    _summary_

    Args:
        folder_name (_type_): _description_
        
    Returns:
        type
    """
    
    
def validate_image(image_path):
    """
    Validates an image file by checking its file type and file readability,
    and re-encoding the image to detect potential issues.
    Image file is removed if it fails any of these tests.

    Args:
        image_path (str): Path to the image file.
        
    Raises:
        Exception: If an error occurs while reading the file.
    """
    
    image_exts = {"jpeg", "jpg", "bmp", "png"}
    
    try:  
        # Check file type using filetype library
        kind = filetype.guess(image_path)

        if not kind or kind.extension not in image_exts:
            print(f"Image not in the ext list: {image_path}")
            os.remove(image_path)
            return

        # Load image to ensure its readable
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Unreadable image: {image_path}")
            os.remove(image_path)
            return
            
        # Force re-encoding to detect potential issues
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            print(f"Corrupt JPEG data detected during re-encoding: {image_path}")
            os.remove(image_path)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        os.remove(image_path)


def clean_dataset(data_dir):
    """
    Cleans a dataset directory by validating and removing corrupted/invalid images.

    Args:
        data_dir (str): Path to the root directory of the dataset.
                        Each subdirectory should correspond to a class of images
    """
    
    # ThreadPoolExecutor is used to parallelise the process
    with ThreadPoolExecutor() as executor:
        for image_class in os.listdir(data_dir):
            class_path = os.path.join(data_dir, image_class)
            
            # Validates each image using validate_image function
            for image in os.listdir(class_path):
                image_path = os.path.join(class_path, image)
                executor.submit(validate_image, image_path)
    

# Main execution
if __name__ == "__main__":
    clean_dataset(Config.RAW_DATA_DIR)
    
    
    
    