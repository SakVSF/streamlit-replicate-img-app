import numpy as np
import onnxruntime as rt
from skimage import io
from pathlib import Path
from PIL import Image
import random


#Please double check this function, since i have a different file structure, i keep getting the error :  FileNotFoundError: The directory 'C:\\images' does not exist

def save_image_to_folder(save_folder, uploaded_image_name, image_array):
                        save_path = Path(save_folder) / uploaded_image_name
                        io.imsave(save_path, image_array)
                        print(f"Image saved successfully at: {save_path}")




#This is to get the single image prediction saved
def get_gan_prediction(input_path, model_path):
                    max_dim = 512

                    def load_and_preprocess_image(image_path, target_size=(max_dim, max_dim, 3)):
                        img = io.imread(image_path)
                        if target_size[0] is not None and target_size[1] is not None:
                            img = np.array(Image.fromarray(img).resize((target_size[1], target_size[0])))
                        img_array = img.astype(np.float32) / 255.0
                        return img_array

                    content_image = load_and_preprocess_image(input_path)
                    content_image = (np.expand_dims(content_image, axis=0))

                    x = content_image

                    providers = ['CPUExecutionProvider']
                    m = rt.InferenceSession(model_path, providers=providers)
                    onnx_pred = m.run(None, {"input": x})[0][0]

                    onnx_pred = (onnx_pred - onnx_pred.min()) / (onnx_pred.max() - onnx_pred.min())
                    onnx_pred = (onnx_pred * 255).astype(np.uint8)

                    print(np.array(onnx_pred).shape)

                    #output_path = 'onnx_generated_city.jpg'    

                    #io.imsave(output_image_path, onnx_pred)

                  
                    #save_image_to_folder(save_folder, uploaded_image_name, image_to_save)
                    return onnx_pred

#example usage
#get_gan_prediction("input_path", "generator_block5_conv2.onnx")


def combine_images_random_block(input_path, block_size=4):
    

    image1 = get_gan_prediction(input_path, "block3conv2.onnx")
    image2 = get_gan_prediction(input_path, "block3conv3.onnx")
    image3 = get_gan_prediction(input_path, "block5conv4.onnx")
    image4 = get_gan_prediction(input_path, "block5conv3.onnx")
    
    # Load images and normalize
    images = np.array([image1,image2,image3,image4])
    image_height, image_width, channels = images[0].shape
    num_images = len(images)
    print(num_images)
    
    # Initialize combined image array
    combined_image = np.zeros_like(images[0], dtype=np.uint8)

    # Iterate over each block of pixels in the output image
    for y in range(0, image_height, block_size):
        for x in range(0, image_width, block_size):
            # Randomly choose one of the input images
            random_image_index = random.randint(0, num_images - 1)
            # Assign the block of pixels from the randomly chosen image
            combined_image[y:y+block_size, x:x+block_size, :] = images[random_image_index][y:y+block_size, x:x+block_size, :]
    
    #save_folder = '/images'
    #uploaded_image_name = 'output_random_block_pixels.jpg'
    #save_image_to_folder(save_folder, uploaded_image_name, combined_image)  #commented so that this can be fixed later
    #io.imsave(output_image_path, combined_image)
    return combined_image



#example usage
#output_image = combine_images_random_block('input_path', block_size=8)




