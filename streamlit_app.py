import glob
import streamlit as st
from streamlit_image_select import image_select
from pathlib import Path
import os
import json
import tempfile

# UI configurations
st.set_page_config(page_title="Neural Style Transfer",
                   page_icon=":bridge_at_night:",
                   layout="wide")

# Placeholder for gallery
home = st.empty()
gen_image = st.empty()
gallery= st.empty()

def show_home(submitted, output_image_path):
    gen_image.empty()
    gallery.empty()
    submitted = False
    
    with home.container():
        st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/023/900/029/non_2x/abstract-colorful-watercolor-background-watercolor-texture-digital-art-painting-illustration-hand-painted-watercolor-abstract-morning-light-wallpaper-it-is-a-hand-drawn-vector.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
        st.markdown("# :rainbow[Automating Visual Artistry]")
        

        with st.form("my_form"):
            
            st.info("**Hello! Bring out your inner Picasso here ↓**", icon="👋🏾")
            with st.expander("**Choose your model**"):
                style = st.selectbox('Style', ('Mosaic', 'VanGogh', 'Picasso', 'GAN-1', 'GAN-2', 'GAN-3', 'GAN-4', 'GAN-5'))
                caption = st.text_input("Write a unique caption", value="My first painting")
                
            uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

            if submitted:
                # Save the uploaded file to a temporary directory
                save_folder = '/workspaces/streamlit-replicate-img-app/gallery'
                # Ensure the folder exists, if not create it
                os.makedirs(save_folder, exist_ok=True)
                
                if uploaded_image1 is not None:
                    save_path1 = Path(save_folder, uploaded_image1.name)
                    with open(save_path1, "wb") as f:
                        f.write(uploaded_image1.getvalue())
                    st.success(f'File {uploaded_image1.name} is successfully saved at {save_path1}')
                    
                    # Generate description for the uploaded image
                    generate_description(uploaded_image1.name, style, caption)
                else:
                    st.error('Error in submitting, please try again')
                    
        if submitted:    
            home.empty() 
            # TODO: A FUNCTION CALL -> call to function that takes in uploaded image at save_path1 and stores output image at some other path
            show_output(uploaded_image1)

def generate_description(image_name, style, caption):
    # Remove file extension from the image name
    image_name_without_extension = os.path.splitext(image_name)[0]
    
    description_file_path = '/workspaces/streamlit-replicate-img-app/gallery/description.json'
    
    # Load existing descriptions if the file exists
    descriptions = {}
    if os.path.exists(description_file_path):
        with open(description_file_path, "r") as desc_file:
            descriptions = json.load(desc_file)
    
    # Update the descriptions with the new image information
    descriptions[image_name_without_extension] = {"model_name": style, "description": caption}
    
    # Write the updated descriptions back to the file
    with open(description_file_path, "w") as desc_file:
        json.dump(descriptions, desc_file, indent=4)

def show_gallery():
    gen_image.empty()
    home.empty()
    
    with gallery.container():
        st.markdown("# :rainbow[Gallery]")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://b.rgbimg.com/users/x/xy/xymonau/600/nLICqmW.jpg");
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # Read all descriptions from the description file
        description_file_path = '/workspaces/streamlit-replicate-img-app/gallery/description.json'
        if os.path.exists(description_file_path):
            with open(description_file_path, "r") as desc_file:
                descriptions = json.loads(desc_file.read())
                print("hi", descriptions)
        else:
            descriptions = {}
        
        # Display all images stored in the local folder
        save_folder = '/workspaces/streamlit-replicate-img-app/gallery'
        paths = [(file_path, file_path.stat().st_mtime) for file_path in Path(save_folder).iterdir()
                 if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        sorted_image_paths = sorted(paths, key=lambda x: x[1], reverse=True)
        image_paths = [str(file_path) for file_path, _ in sorted_image_paths]

        if image_paths:
            n = st.number_input("Grid Width", 1, 5, 2)
            groups = []
            for i in range(0, len(image_paths), n):
                groups.append(image_paths[i:i+n])

            for group in groups:
                cols = st.columns(n)
                for i, image_path in enumerate(group):
                    image_name = Path(image_path).stem
                    # Check if the image_name exists in the descriptions dictionary
                    if image_name in descriptions:
                        model_name = descriptions[image_name].get("model_name", "")
                        description = descriptions[image_name].get("description", "")
                    else:
                        model_name = ""
                        description = ""
                    
                    # Construct caption including model name and optional description
                    caption_parts = [f"{image_name} - Model: {model_name}"]
                    if description:
                        caption_parts.append(f"Description: {description}")
                    
                    caption = " - ".join(caption_parts).title()
                    
                    # Display the image with its caption
                    cols[i].image(image_path, caption=caption, use_column_width=True)

        else:
            # Display a message if no images are found in the gallery folder
            st.write("No images available in the gallery.")

def show_output(output):

    gallery.empty()
    home.empty()
  
    with gen_image.container():
        st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://t3.ftcdn.net/jpg/06/27/85/70/360_F_627857047_VDETCsSfRkZuo5Fzdy3eHL1ZGdhsqYv9.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
        st.markdown("# :rainbow[Automating Visual Artistry]")
        
       
        st.image(output, caption="Generated Image 🎈", use_column_width=True)

        #VIEW IN GALLERY BUTTON NOT WORKING-> SHOULD ROUTE TO GALLERY ON CLICKING 
        #view_in_gallery = st.button("View in Gallery", use_container_width=True)

    #if view_in_gallery:
        #st.sidebar.radio("Navigation", ("Home", "Gallery"), index=1)
        #show_gallery()
    # Update the selected page in the sidebar to "Gallery"

def main():
    submitted = False

    output_image_path = None
    page = st.sidebar.radio("Navigation", ("Home", "Gallery"))

    if page == "Home":
        show_home(submitted, output_image_path)
    elif page == "Gallery":
        show_gallery()

if __name__ == "__main__":
    main()
