import cv2 as cv
import numpy as np
from torchvision import transforms
import os

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path, target_shape=None):

    img = cv.imread(img_path)[:, :, ::-1] 
    
    current_height, current_width = img.shape[:2]
    new_height = target_shape
    new_width = int(current_width * (new_height / current_height))
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0  #\
    return img


def normalize_nst_input(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0) 

    return img

def generate_gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram

def save_iteration_output(optimizing_img, dump_path, config, img_id, num_of_iterations):
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2) 

    if True:
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1]
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
