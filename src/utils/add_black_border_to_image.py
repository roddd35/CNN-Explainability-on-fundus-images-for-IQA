import os
from PIL import Image

def make_image_square_with_black_borders_and_save(image_path):
    original_image = Image.open(image_path)
    width, height = original_image.size
    
    new_size = max(width, height)
    bordered_image = Image.new("RGB", (new_size, new_size), "black")
    
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    bordered_image.paste(original_image, (left, top))
    
    bordered_image.save(image_path)
    
    original_image.close()
    bordered_image.close()

def main():
    path = '/home/rodrigocm/scratch/datasets/eyeq/images'
    extensions = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(path):
        if filename.endswith(extensions):
            image_path = os.path.join(path, filename)
            make_image_square_with_black_borders_and_save(image_path)

main()
