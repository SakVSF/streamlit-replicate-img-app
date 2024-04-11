import streamlit as st
from streamlit_image_select import image_select
from pathlib import Path
import os
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
    submitted=False

    
    
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
                save_folder = 'C:/Users/Saakshi Saraf/Downloads/test'
                # Ensure the folder exists, if not create it
                os.makedirs(save_folder, exist_ok=True)
                
                if uploaded_image1 is not None:
                    save_path1 = Path(save_folder, uploaded_image1.name)
                    with open(save_path1, "wb") as f:
                        f.write(uploaded_image1.getvalue())
                    st.success(f'File {uploaded_image1.name} is successfully saved at {save_path1}')     

                else:
                    st.error('Error in submitting, please try again')
                    
        if submitted:    
            home.empty() 
            #TODO: A FUNCTION CALL -> call to function that takes in uploaded image at save_path1 and stores output image at some other path
            show_output(uploaded_image1)

        
        

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
        
        #Display all images stored in local folder
        save_folder = 'C:/Users/Saakshi Saraf/Downloads/test'
        paths = [(file_path, file_path.stat().st_mtime) for file_path in Path(save_folder).iterdir()
                 if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        sorted_image_paths = sorted(paths, key=lambda x: x[1], reverse=True)
        image_paths = [str(file_path) for file_path, _ in sorted_image_paths]
    
        img = image_select(
            label="A collection of your artpieces, delivered through our Neural Style Transfer models!",
            images=image_paths,
            use_container_width=True
        )


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
