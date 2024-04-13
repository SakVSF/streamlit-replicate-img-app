import os
import argparse
import torch
import utils.utils as utils
from models.transformer_net import ImageTransfomer

def stylize_static_image(inference_config):
    """
    Stylizes a static image using a neural style transfer model.

    Parameters:
        inference_config (dict): Dictionary containing inference configuration parameters.

    Returns:
        str: Path to the stylized image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nst_model = ImageTransfomer.from_pretrained("SavageSanta25/johnson-mosaic").to(device)
    print(os.path.join(inference_config["model_binaries_path"], inference_config["checkpoint_name"]))
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["checkpoint_name"]))
    print(training_state.keys())
    state_dict = training_state["state_dict"]
    nst_model.load_state_dict(state_dict, strict=True)
    nst_model.eval()

    # Stylize using trained model
    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = nst_model(content_image).to('cpu').numpy()[0]
        output_image_path = os.path.join(inference_config['output_images_path'], f"stylized_{inference_config['content']}")
        utils.save_and_maybe_display_image(output_image_path, stylized_img, should_display=inference_config['should_not_display'])

    return output_image_path