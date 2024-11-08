from image_processor import ImageProcessor
from PIL import Image
import numpy as np
import os
import glob

def get_image_paths(folder_path, extensions=["*.jpg", "*.png", "*.jpeg"]):
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_paths

def save_image(image, file_path):
    image.save(file_path)
    print(f"Image saved to {file_path}")

folder_path = "SOTS/hazy"
image_paths = get_image_paths(folder_path)
psnr_values = []

for image_path in image_paths:
    clear_image_path = "SOTS/gt/" + image_path[10:14] + ".png"
    clear_image = Image.open(clear_image_path)
    new_clear_image_path = "SOTS/gt1/" + image_path[10:14] + ".png"
    clear_image.save(new_clear_image_path)
    print(clear_image_path)
    image = Image.open(image_path)
    new_path = "SOTS/hazy1/" + image_path[10:14] + ".png"
    image.save(new_path)

    processor = ImageProcessor(image_path=image_path, patch_size=3, beta=1)
    enhanced_image = processor.enhance_image()
    enhanced_path = "SOTS/enhanced/" + image_path[10:14] + ".jpg"
    print(enhanced_path)
    save_image(enhanced_image, enhanced_path)
    enhanced_image_np = np.array(enhanced_image)
    enhanced_image.show()

    clear_image = Image.open(clear_image_path)
    clear_image_np = np.array(clear_image)

    mse = np.mean((enhanced_image_np - clear_image_np) ** 2)
    psnr_value = 20 * np.log10(255 / np.sqrt(mse))
    psnr_values.append(psnr_value)

print(psnr_values)
np.save("psnr_values.npy", psnr_values)

mean_psnr = np.mean(np.array(psnr_values))
np.save("mean_psnr_value.npy", mean_psnr)
print("Mean:", mean_psnr)