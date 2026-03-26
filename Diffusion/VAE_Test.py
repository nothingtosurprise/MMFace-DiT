import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
# Path to the local VAE directory containing the model and config.json
VAE_PATH = "VAE" 
# Path to input image
IMAGE_PATH = "/mnt/e/Multi-Modal Face Generation/Celeb_Dataset/Celeb_Final/test/images/5.jpg"
# Directory where the output plot will be saved
OUTPUT_DIR = "Test_Gens/VAE"
# The resolution to process the image at.
IMAGE_SIZE = 512

def main():
    """
    Main function to load the VAE, process an image, and save the results.
    """
    print("--- VAE Reconstruction Test Script ---")

    # --- 1. Setup Environment ---
    if not os.path.exists(VAE_PATH):
        print(f"Error: VAE directory not found at '{VAE_PATH}'.")
        print("Please ensure the script is in the same directory as the 'VAE' folder.")
        return
        
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Input image not found at '{IMAGE_PATH}'.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # Set the device (use GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use bfloat16 for better performance if available on CUDA
    weight_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # --- 2. Load the FLUX VAE ---
    print(f"Loading VAE from local path: '{VAE_PATH}'...")
    try:
        vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device=device, dtype=weight_dtype)
        print("VAE loaded successfully.")
    except Exception as e:
        print(f"Failed to load VAE. Error: {e}")
        return

    # --- 3. Load and Preprocess the Image ---
    print(f"Loading and preprocessing image: '{os.path.basename(IMAGE_PATH)}'")
    
    # Define the transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Normalize pixels to [-1, 1]
    ])

    # Load the original image for plotting
    original_image_pil = Image.open(IMAGE_PATH).convert("RGB")
    
    # Preprocess the image for the model
    image_tensor = preprocess(original_image_pil).unsqueeze(0).to(device=device, dtype=weight_dtype)
    print(f"Image tensor prepared with shape: {image_tensor.shape}")

    # --- 4. Encode the Image into the Latent Space ---
    print("Encoding image to latent representation...")
    with torch.no_grad():
        # The VAE encoder returns a distribution object
        latent_dist = vae.encode(image_tensor).latent_dist
        # Sample from the distribution to get the latent tensor
        latents = latent_dist.sample()
        
    print(f"Generated latents with shape: {latents.shape}") # Should be [1, 16, H/8, W/8]

    # --- 5. Decode the Latent Representation back to an Image ---
    print("Decoding latents back to image...")
    with torch.no_grad():
        # The decode method returns a tensor with pixel values in [-1, 1]
        decoded_tensor = vae.decode(latents).sample[0] # Get the first image from the batch

    # --- 6. Post-process for Visualization ---
    latent_for_plot = latents[0].mean(dim=0).detach().cpu().to(torch.float32).numpy()
    latent_for_plot = (latent_for_plot - latent_for_plot.min()) / (latent_for_plot.max() - latent_for_plot.min())

    # Decoded image: Convert tensor back to a PIL image
    decoded_tensor = (decoded_tensor / 2 + 0.5).clamp(0, 1) # Denormalize to [0, 1]
    decoded_image_pil = transforms.ToPILImage()(decoded_tensor.detach().cpu().to(torch.float32))

    # --- 7. Plot and Save the Results ---
    print("Generating and saving comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle("FLUX.1-Krea-dev VAE Reconstruction", fontsize=16)

    # Plot Original Image
    axes[0].imshow(original_image_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')

    # Plot Latent Visualization
    axes[1].imshow(latent_for_plot, cmap='gray')
    axes[1].set_title(f"2. Latent Representation (16 channels averaged)")
    axes[1].axis('off')
    
    # Plot Decoded Image
    axes[2].imshow(decoded_image_pil)
    axes[2].set_title("3. Decoded Image")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, "vae_reconstruction_test.png")
    plt.savefig(output_path)
    
    print("-" * 30)
    print(f"✅ Success! Plot saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()