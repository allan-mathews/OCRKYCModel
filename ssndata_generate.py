# Read libraries here
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from imgaug import augmenters as iaa
import imageio.v2 as imageio
import cv2


# Function to generate ssn card images
def generate_cardssn(id, name):
    """_Generate ssn card template using Slab-Serif.ttf as the font_
    Args:
        id (_string_): _9 digit SSN id in the form 'DDD-DD-DDDD'_
        name (_STRING_): _Name shown in the ssn id_

    Returns:
        _image_object_: _SSN Card image without any enhancements_
    """
    template = Image.open("template.png")
    font = ImageFont.truetype("Slab-Serif.ttf", size=25)
    draw = ImageDraw.Draw(template)
    draw.text((250, 180), id, font=font, fill='black', align='left')
    draw.text((350, 260), name, font=font,
              fill='black', align="center", anchor="mm")
    return template


def generate_aug_ssn(record):
    """To Generate augmeneted SSN images
    ## Image augmentation
    Below enhancements are applied
    -   Cutout (Randomly fill the image with white squares)
    -   Dropout (Randomly set pixels to zero)
    -   Gaussian Blur / Median Blur
    -   Additive Gaussian Noise
    -   Rotation (Skew)
    -   Perspective Transform
    -   Multiply (Changes the brightness of the image)

    Args:
        record (_list_): _ssn records loaded from a csv dataset_

    Returns:
        _list_: _augmented images_
    """
    card = generate_cardssn(record['ID'], record['Name'])
    image = np.asarray(card)
    # 32 means create 32 enhanced images using following methods.
    images = np.array([image for _ in range(32)], dtype=np.uint8)
    # Image Augmentation done here
    seq = iaa.Sequential([
        # 30%of the images has one cutout atmost
        iaa.Sometimes(0.2, iaa.Cutout(nb_iterations=1, seed=1)),
        iaa.Dropout(p=(0, 0.1), seed=1),  # gives grains to the picture
        # blurs the image using gaussian function
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
        iaa.Flipud(0.4),  # half of the images are flipped vertically
        # 30% of the images is skewed to a maximum of 30 deg
        iaa.Sometimes(0.3, iaa.Rotate((-30, 30))),
        iaa.PerspectiveTransform(
            scale=(0, 0.1)),  # scales the image maximum 10%
        iaa.Multiply((0.5, 1.5)),  # change the brghtness

    ])
    augmented_images = seq(images=images)
    return augmented_images


def generate_images():
    records = pd.read_csv('sssn_data.csv').to_dict(orient='records')
    # records, range(len(records)):
    for record, i in zip(records[:3], range(3)):
        augmented_images = generate_aug_ssn(record)
        for augmented_image, j in zip(augmented_images, range(32)):
            # write all changed images
            imageio.imwrite(
                f'Data/ {str(i)}aug{str(j)}.PNG', augmented_images[j])
    print('Success and done')


# generate_images()
