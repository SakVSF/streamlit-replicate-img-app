import streamlit as st
import requests
import zipfile
#import io
from utils import icon
from pathlib import Path
import os
from PIL import Image
import numpy as np
import onnxruntime as rt
from skimage import io
from skimage import img_as_ubyte




# UI configurations
st.set_page_config(page_title="Neural Style Transfer",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon()
st.markdown("# :rainbow[Automating Visual Artistry]")
# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()
def show_icon():
    """Shows a gallery button."""

    st.markdown("""
    <style>
    .gallery-button {
        padding: 0.5rem 1rem;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<button onclick="show_gallery()" class="gallery-button">Gallery</button>', unsafe_allow_html=True)

    st.markdown('<script>function show_gallery() {window.location.href = "#gallery";}</script>', unsafe_allow_html=True)

def show_navbar():
    # Navigation bar with a gallery button
    st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 1rem;
        background-color: #f0f0f0;
    }
    .gallery-button {
        padding: 0.5rem 1rem;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<a href="#gallery" class="gallery-button">Gallery</a>', unsafe_allow_html=True)


def show_gallery():
    # Display the gallery
    st.markdown("<h2 id='gallery'>Gallery</h2>", unsafe_allow_html=True)



def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Hello! Start here ↓**", icon="👋🏾")
            with st.expander(":rainbow[**Choose your model here**]"):
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Width of output image", value=1024)
                height = st.number_input("Height of output image", value=1024)
                num_outputs = st.slider(
                    "Number of images to output", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                       'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Number of denoising steps", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)", value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
                
                #TODO: create dropdown of the models 

            #TODO : 
            
            ##################################
            uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
            uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)
            
            if submitted:
                # Save uploaded file to 'F:/tmp' folder.
                save_folder = 'F:/tmp'
                save_path = Path(save_folder, uploaded_image1.name)
                os.write(1, f"{save_path}\n".encode())

                def get_prediction(input_path, model_path):
                    max_dim = 512

                    def load_and_preprocess_image(image_path, target_size=(max_dim, max_dim, 3)):
                        img = io.imread(image_path)
                        if target_size[0] is not None and target_size[1] is not None:
                            img = np.array(Image.fromarray(img).resize((target_size[1], target_size[0])))
                        img_array = img.astype(np.float32) / 255.0
                        return img_array

                    #content_image = load_and_preprocess_image(input_path)
                    content_image = load_and_preprocess_image('city.jpg')

                    content_image = (np.expand_dims(content_image, axis=0))

                    x = content_image

                    model_path = 'generator_block5_conv2.onnx'
                    #model_path = model_path

                    providers = ['CPUExecutionProvider']
                    m = rt.InferenceSession(model_path, providers=providers)
                    onnx_pred = m.run(None, {"input": x})[0][0]

                    onnx_pred = (onnx_pred - onnx_pred.min()) / (onnx_pred.max() - onnx_pred.min())
                    onnx_pred = (onnx_pred * 255).astype(np.uint8)



                    print(np.array(onnx_pred).shape)

                    output_path = 'onnx_generated_city.jpg'    

                    io.imsave(output_path, onnx_pred)

                    def save_image_to_folder(save_folder, uploaded_image_name, image_array):
                        save_path = Path(save_folder) / uploaded_image_name
                        io.imsave(save_path, image_array)
                        print(f"Image saved successfully at: {save_path}")

                    save_folder = 'F:/tmp'  # Specify the path to your desired output folder
                    uploaded_image_name = 'uploaded_image.jpg'  #uploaded_image1.name

                    image_to_save = onnx_pred  # Your image array to save

                    save_image_to_folder(save_folder, uploaded_image_name, image_to_save)

                    st.image('onnx_generated_city.jpg')

            


            # model_selection = None
            # if model_selection == Gan1:
            #     get_prediction(input_path, Gan1_model_path)
                
            # if model_selection == Gan2:
            #     get_prediction(input_path, Gan2_model_path)


        #########################################
                if save_path.exists():
                    st.success(f'File {uploaded_image1.name} is successfully saved!')

    

        return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, uploaded_image1, uploaded_image2


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  
                        # List to store all generated images
                        # output = replicate.run(
                        #     REPLICATE_MODEL_ENDPOINTSTABILITY,
                        #     input={
                        #         "prompt": prompt,
                        #         "width": width,
                        #         "height": height,
                        #         "num_outputs": num_outputs,
                        #         "scheduler": scheduler,
                        #         "num_inference_steps": num_inference_steps,
                        #         "guidance_scale": guidance_scale,
                        #         "prompt_stregth": prompt_strength,
                        #         "refine": refine,
                        #         "high_noise_frac": high_noise_frac
                        #     }
                        # )
                        output = img = Image.open("image.jpg")

                        if output:
                            st.toast(
                                'Your image has been generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Generated Image 🎈",
                                             use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Download All Images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Images generated!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass





def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar()
    main_page(submitted, width, height, num_outputs, scheduler, num_inference_steps,
              guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt)


if __name__ == "__main__":
    main()
