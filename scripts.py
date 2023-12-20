import xml.etree.ElementTree as ET
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from tqdm import tqdm
import random
import numpy as np
import os

path = os.path.abspath(__file__).split('\\')
path.pop(len(path) - 1)
base_dir = '\\'.join(path)


def rewrite_annotations(dir_path):
    for filename in os.listdir(dir_path):
        cur_path = f"{dir_path}\\{filename}"
        tree = ET.parse(cur_path)
        root = tree.getroot()
        size = root.find('size')
        size_width = size.find('width').text
        size_height = size.find('height').text
        size_depth = size.find('depth').text
        object_bndbox = root.find('object').find('bndbox')
        object_name = root.find('object').find('name').text
        object_bndbox_xmin = object_bndbox.find('xmin').text
        object_bndbox_ymin = object_bndbox.find('ymin').text
        object_bndbox_xmax = object_bndbox.find('xmax').text
        object_bndbox_ymax = object_bndbox.find('ymax').text
        new_filename = f"{filename[:-3]}jpg"
        xml_text = f"""
<annotation>
    <folder>images</folder>
    <filename>{new_filename}</filename>
    <size>
        <width>{size_width}</width>
        <height>{size_height}</height>
        <depth>{size_depth}</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>{object_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{object_bndbox_xmin}</xmin>
            <ymin>{object_bndbox_ymin}</ymin>
            <xmax>{object_bndbox_xmax}</xmax>
            <ymax>{object_bndbox_ymax}</ymax>
        </bndbox>
    </object>
</annotation>"""
        with open(cur_path, "w") as f:
            f.writelines(xml_text)


def augment_images(dataset_path, mode="train", expected_images_num=100):
    def add_noise(img):
        deviation = 10 * random.random()
        noise = np.random.normal(0, deviation, img.shape)
        img += noise
        np.clip(img, 0., 255.)
        return img

    categories = os.listdir('{}/{}'.format(dataset_path, mode))
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

    for category in tqdm(categories, total=len(categories), desc='Augmenting Images'):
        num_images = len(os.listdir(r'{}\{}\{}'.format(dataset_path, mode, category)))
        if num_images < expected_images_num:
            images_to_augment = os.listdir(r'{}\{}\{}'.format(dataset_path, mode, category))
            num_augments_per_image = (expected_images_num - num_images) / num_images
            if num_augments_per_image == 0:
                images_to_augment = np.random.choice(images_to_augment, (expected_images_num - num_images))
                num_augments_per_image = 1
            for image_path in images_to_augment:
                image = load_img(r'{}\{}\{}\{}'.format(dataset_path, mode, category, image_path))
                x = img_to_array(image)
                x = x.reshape((1,) + x.shape)
                x = add_noise(x)
                i = 0
                for _ in datagen.flow(x, batch_size=1,
                                      save_to_dir=r'{}\{}\{}'.format(dataset_path, mode, category),
                                      save_prefix=image_path[:-4], save_format='jpg'):
                    i += 1
                    if i > num_augments_per_image:
                        break
    print('Done augmenting images!')


if __name__ == '__main__':
    #dirr = r'C:\Users\artyo\Desktop\imgs\train'
    #augment_images(r'C:\Users\artyo\PycharmProjects\rcnn\new_eval_data', 'train', 100)
    dirr = r"C:\Users\artyo\PycharmProjects\rcnn\eval_data_2\annotations"
    k = 0
    rewrite_annotations(dirr)
