# Import necessary libraries 
import os
import cv2
import magic
from concurrent.futures import ThreadPoolExecutor
from config import Config

def validate_image(image_path):
    """
    Validates an image file by checking its MIME type using python-magic.
    Only allows JPEG and JPG files. Removes the file if it is invalid.
    Also checks if the image can be successfully loaded using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Raises:
        Exception: If an error occurs while reading the file.
    """
    try:
        # Use python-magic to check MIME type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(image_path)

        if file_type not in ["image/jpeg"]:
            print(f"Invalid file type: {image_path} ({file_type})")
            os.remove(image_path)
            return False

        # Check if the image can be loaded using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unreadable image: {image_path}")
            os.remove(image_path)
            return False

        return True

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        os.remove(image_path)
        return False

def clean_dataset(data_dir):
    """
    Cleans a dataset directory by validating and removing invalid images.

    Args:
        data_dir (str): Path to the root directory of the dataset.
                        Each subdirectory should correspond to a class of images
    """
    validated_count = 0
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for image_class in os.listdir(data_dir):
            class_path = os.path.join(data_dir, image_class)

            # Submit each image validation task to the executor
            for image in os.listdir(class_path):
                image_path = os.path.join(class_path, image)
                futures.append(executor.submit(validate_image, image_path))

        for future in futures:
            if future.result():
                validated_count += 1

    print(f"Total validated files: {validated_count}")

# Main execution
if __name__ == "__main__":
    clean_dataset(Config.RAW_DATA_DIR)
