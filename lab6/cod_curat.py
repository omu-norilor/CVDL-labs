import gradio as gr
from PIL import Image
import torch

from library import UNet

def load_Unet():
    model = UNet(retain_dim=True, out_sz=(250, 250), num_class=3)
    model.load_state_dict(torch.load('unet.pth'))
    model.eval()

    # Dummy input for tracing (adjust the size based on your input size)
    dummy_input = torch.randn(1, 3, 250, 250).cuda()
    scripted_model = torch.jit.script(model)
    scripted_model.save("scripted_unet.pt")

def predict_segmentation(image):
    # Preprocess the input image
    # You might need to adjust the preprocessing based on your original training pipeline
    image = Image.fromarray(image.astype('uint8'))
    # Perform any necessary image preprocessing (resizing, normalization, etc.)
    # ...

    # Convert the image to a PyTorch tensor
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Post-process the output (e.g., convert to segmentation mask)
    # ...

    return output.squeeze().cpu().numpy()



if __name__ == '__main__':
    load_Unet()
    # Load the TorchScript model
    model = torch.jit.load("scripted_unet.pt")
    model.eval()

    # Gradio Interface
    iface = gr.Interface(
        fn=predict_segmentation,
        inputs="image",
        outputs="image",
        live=True,
        interpretation="default"
    )

    iface.launch()
