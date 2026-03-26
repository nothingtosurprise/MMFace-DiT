# utils_final.py
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
import random
from pathlib import Path

logger_utils = logging.getLogger(__name__)

# --- Utility Functions ---
def ensure_rgb(img: Image.Image) -> Image.Image:
    """Ensures an image is in RGB mode."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def ensure_grayscale(img: Image.Image) -> Image.Image:
    """Ensures an image is in grayscale mode."""
    if img.mode != "L":
        return img.convert("L")
    return img

# --- Unified Dataset Classes ---

class PrecomputedUnifiedLatentDataset(Dataset):
    """
    Unified Dataset class for pre-computed image latents, mask latents, sketch latents, and text captions.
    Loads pre-computed latents from a structured directory hierarchy.
    Directory structure:
    latents_root/
        Celeb/
            image_latents/
                00000.pt
            mask_latents/
                00000.pt
            sketch_latents/
                00000.pt
            text/
                00000.txt
        FFHQ/
            image_latents/
                00000.pt
            mask_latents/
                00000.pt
            sketch_latents/
                00000.pt
            text/
                00000.txt
    """
    def __init__(self, latents_root, tokenizer,
                 text_dropout_prob=0.0, mask_dropout_prob=0.0, sketch_dropout_prob=0.0):
        # --- Store Paths and Configuration ---
        self.latents_root = Path(latents_root)
        self.tokenizer = tokenizer
        self.text_dropout_prob = text_dropout_prob
        self.mask_dropout_prob = mask_dropout_prob
        self.sketch_dropout_prob = sketch_dropout_prob
        # --- Discover and Validate Data ---
        self.data_samples = self._discover_data_samples()
        logger_utils.info(f"Discovered {len(self.data_samples)} total samples across all datasets")

    def _discover_data_samples(self):
        """Discover all valid data samples in the structured directory hierarchy."""
        data_samples = []
        # Check if latents root exists
        if not self.latents_root.exists():
            raise FileNotFoundError(f"Latents root directory not found: {self.latents_root}")
        # Iterate through dataset subdirectories (Celeb, FFHQ, etc.)
        for dataset_dir in self.latents_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            image_latents_dir = dataset_dir / "image_latents"
            mask_latents_dir = dataset_dir / "mask_latents"
            sketch_latents_dir = dataset_dir / "sketch_latents"
            text_dir = dataset_dir / "text"
            # Validate directory structure
            if not all([image_latents_dir.exists(), mask_latents_dir.exists(), sketch_latents_dir.exists(), text_dir.exists()]):
                logger_utils.warning(f"Incomplete directory structure for dataset {dataset_name}. Skipping...")
                continue
            # Find all .pt files in image_latents directory
            image_latent_files = list(image_latents_dir.glob("*.pt"))
            logger_utils.info(f"Found {len(image_latent_files)} image latent files in {dataset_name}")
            # For each image latent, check if corresponding mask, sketch latent and text file exist
            # ASSUMPTION: All files exist, skip verification for efficiency
            for img_latent_path in image_latent_files:
                base_name = img_latent_path.stem
                mask_latent_path = mask_latents_dir / f"{base_name}.pt"
                sketch_latent_path = sketch_latents_dir / f"{base_name}.pt"
                text_path = text_dir / f"{base_name}.txt"

                data_samples.append({
                    'dataset': dataset_name,
                    'base_name': base_name,
                    'image_latent_path': str(img_latent_path),
                    'mask_latent_path': str(mask_latent_path),
                    'sketch_latent_path': str(sketch_latent_path),
                    'text_path': str(text_path)
                })

        return data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        try:
            # --- Load Pre-computed Image Latent ---
            image_latent = torch.load(sample['image_latent_path'])
            image_latent = image_latent.squeeze().to(dtype=torch.float32)

            # --- Randomly Choose Modality ---
            use_mask = random.choice([True, False])
            modality_flag = torch.tensor(0) if use_mask else torch.tensor(1) # 0 for mask, 1 for sketch

            # --- Load and Process Conditioning Latent ---
            if use_mask:
                conditioning_latent = torch.load(sample['mask_latent_path'])
                dropout_prob = self.mask_dropout_prob
                log_msg = "mask"
            else: # use sketch
                conditioning_latent = torch.load(sample['sketch_latent_path'])
                dropout_prob = self.sketch_dropout_prob
                log_msg = "sketch"

            conditioning_latent = conditioning_latent.squeeze().to(dtype=torch.float32)

            # --- Apply Modality Dropout ---
            if random.random() < dropout_prob:
                conditioning_latent = torch.zeros_like(conditioning_latent)
                logger_utils.debug(f"Dropped {log_msg} for sample {sample['base_name']} from {sample['dataset']} (Index {idx})")

            # --- Load and Process Text (with potential dropout) ---
            if random.random() < self.text_dropout_prob:
                chosen_caption = ""  # Use empty string for unconditional/dropped text
                logger_utils.debug(f"Dropped text for sample {sample['base_name']} from {sample['dataset']} (Index {idx})")
            else:
                try:
                    with open(sample['text_path'], 'r', encoding='utf-8') as f:
                        captions = [line.strip() for line in f if line.strip()]
                    if not captions:
                        logger_utils.warning(f"Text file {sample['text_path']} is empty or contains only whitespace. Using empty caption.")
                        chosen_caption = ""
                    else:
                        chosen_caption = random.choice(captions)
                except UnicodeDecodeError:
                    logger_utils.error(f"Unicode decode error for text file {sample['text_path']}. Using empty caption.")
                    chosen_caption = ""
                except Exception as e:
                    logger_utils.error(f"Error reading text file {sample['text_path']}: {e}. Using empty caption.")
                    chosen_caption = ""

            # --- Tokenize Text with CLIP Tokenizer ---
            clip_tokenized_output = self.tokenizer(
                chosen_caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            clip_input_ids = clip_tokenized_output.input_ids.squeeze(0)

            # Return unified data structure
            return {
                "image_latent": image_latent,
                "conditioning_latent": conditioning_latent,
                "clip_input_ids": clip_input_ids,
                "dataset": sample['dataset'],
                "base_name": sample['base_name'],
                "modality": modality_flag # 0 for mask, 1 for sketch
            }
        except Exception as e:
            logger_utils.error(f"Error loading or processing data at index {idx} (Dataset: {sample['dataset']}, Base Name: {sample['base_name']}): {e}")
            logger_utils.error(f"Image latent path: {sample['image_latent_path']}")
            logger_utils.error(f"Mask latent path: {sample['mask_latent_path']}")
            logger_utils.error(f"Sketch latent path: {sample['sketch_latent_path']}")
            logger_utils.error(f"Text path: {sample['text_path']}")
            raise e


class UnifiedImageDataset(Dataset):
    """
    Unified Dataset class for CelebA-HQ/FFHQ images, masks, sketches, and text captions.
    Includes modality dropout for text, masks, and sketches.
    Supports two dataset structures:
    1. CelebA-HQ style (default/dataset_type='celeba'):
       data_root/
         train/
           images/
             00000.png
           masks/ (or mask_folder)
             00000.png
           sketches/ (or sketch_folder)
             00000.png
       text_root/
         00000.txt
    2. FFHQ style (dataset_type='ffhq'):
       data_root/ (points to the directory containing subdirs, e.g., E:/Datasets/FFHQ/images1024x1024/images1024x1024)
         00000/
           00000.png
         01000/
           01000.png
           ...
       mask_data_root/ (points to the flat directory, e.g., E:/Datasets/FFHQ/Masks_Colored_1024)
         00000.png
         00001.png
         ...
       sketch_data_root/ (points to the flat directory, e.g., E:/Datasets/FFHQ/sketches)
         00000.png
         00001.png
         ...
       text_root/ (points to the flat directory, e.g., E:/Datasets/FFHQ/FFHQ_Captions_Final)
         00000.txt
         00001.txt
         ...
    """
    def __init__(self, data_root, text_root, tokenizer, split='train', resolution=512,
                 image_folder="images", mask_folder="masks", sketch_folder="sketches",
                 mask_suffix=".png", sketch_suffix=".png", text_suffix=".txt",
                 text_dropout_prob=0.0, mask_dropout_prob=0.0, sketch_dropout_prob=0.0,
                 dataset_type='celeba', mask_data_root=None, sketch_data_root=None):
        # --- Store Paths and Configuration ---
        self.data_root = data_root
        self.mask_data_root = mask_data_root
        self.sketch_data_root = sketch_data_root
        self.text_data_root = text_root
        self.split = split
        self.resolution = resolution
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.sketch_folder = sketch_folder
        self.mask_suffix = mask_suffix
        self.sketch_suffix = sketch_suffix
        self.text_suffix = text_suffix
        self.tokenizer = tokenizer
        self.text_dropout_prob = text_dropout_prob
        self.mask_dropout_prob = mask_dropout_prob
        self.sketch_dropout_prob = sketch_dropout_prob
        self.dataset_type = dataset_type.lower()

        if self.dataset_type not in ['celeba', 'ffhq']:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'celeba' or 'ffhq'.")

        # --- Determine File Paths Based on Dataset Type ---
        # ASSUMPTION: All files exist, skip verification for efficiency
        self.image_paths, self.mask_paths, self.sketch_paths, self.text_paths, self.base_names = self._get_file_paths()

        # --- Transform Pipeline ---
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
        self.image_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.sketch_norm = lambda x: (x - 0.5) * 2.0  # Manual normalization to [-1, 1]

    def _get_file_paths_for_celeba(self):
        """Get file paths for CelebA-HQ style structure."""
        img_mask_sketch_root_dir = os.path.join(self.data_root, self.split)
        image_paths_jpg = sorted(glob.glob(os.path.join(img_mask_sketch_root_dir, self.image_folder, "*.jpg")))
        image_paths_png = sorted(glob.glob(os.path.join(img_mask_sketch_root_dir, self.image_folder, "*.png")))
        image_paths = sorted(list(set(image_paths_jpg + image_paths_png)))

        valid_image_paths = []
        mask_paths = []
        sketch_paths = []
        text_paths = []
        base_names = []

        logger_utils.info(f"[CelebA-HQ] Assuming all triplets exist in Img/Mask/Sketch root: {img_mask_sketch_root_dir}, Text root: {self.text_data_root}")
        logger_utils.info(f"[CelebA-HQ] Found {len(image_paths)} potential image files (.jpg or .png).")

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(img_mask_sketch_root_dir, self.mask_folder, f"{base_name}{self.mask_suffix}")
            sketch_path = os.path.join(img_mask_sketch_root_dir, self.sketch_folder, f"{base_name}{self.sketch_suffix}")
            text_path = os.path.join(self.text_data_root, f"{base_name}{self.text_suffix}")

            # ASSUMPTION: All files exist
            valid_image_paths.append(img_path)
            mask_paths.append(mask_path)
            sketch_paths.append(sketch_path)
            text_paths.append(text_path)
            base_names.append(base_name)

        return valid_image_paths, mask_paths, sketch_paths, text_paths, base_names

    def _get_file_paths_for_ffhq(self):
        """Get file paths for FFHQ style structure."""
        if self.mask_data_root is None:
             raise ValueError("mask_data_root must be provided for dataset_type='ffhq'.")
        if self.sketch_data_root is None:
             raise ValueError("sketch_data_root must be provided for dataset_type='ffhq'.")

        image_paths_jpg = sorted(glob.glob(os.path.join(self.data_root, "*", "*.jpg"), recursive=False))
        image_paths_png = sorted(glob.glob(os.path.join(self.data_root, "*", "*.png"), recursive=False))
        image_paths = sorted(list(set(image_paths_jpg + image_paths_png)))

        valid_image_paths = []
        mask_paths = []
        sketch_paths = []
        text_paths = []
        base_names = []

        logger_utils.info(f"[FFHQ] Assuming all triplets exist. Image root: {self.data_root}, Mask root: {self.mask_data_root}, Sketch root: {self.sketch_data_root}, Text root: {self.text_data_root}")
        logger_utils.info(f"[FFHQ] Found {len(image_paths)} potential image files.")

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_data_root, f"{base_name}{self.mask_suffix}")
            sketch_path = os.path.join(self.sketch_data_root, f"{base_name}{self.sketch_suffix}")
            text_path = os.path.join(self.text_data_root, f"{base_name}{self.text_suffix}")

            # ASSUMPTION: All files exist
            valid_image_paths.append(img_path)
            mask_paths.append(mask_path)
            sketch_paths.append(sketch_path)
            text_paths.append(text_path)
            base_names.append(base_name)

        return valid_image_paths, mask_paths, sketch_paths, text_paths, base_names

    def _get_file_paths(self):
        """Dispatch to the correct path discovery method."""
        if self.dataset_type == 'celeba':
            return self._get_file_paths_for_celeba()
        elif self.dataset_type == 'ffhq':
            return self._get_file_paths_for_ffhq()
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        sketch_path = self.sketch_paths[idx]
        text_path = self.text_paths[idx]
        base_name = self.base_names[idx]

        try:
            # --- Load and Transform Image ---
            image = Image.open(img_path)
            image = ensure_rgb(image)
            image_tensor = self.transform(image)
            image_tensor = self.image_norm(image_tensor)

            # --- Randomly Choose Modality ---
            use_mask = random.choice([True, False])
            modality_flag = torch.tensor(0) if use_mask else torch.tensor(1) # 0 for mask, 1 for sketch

            # --- Load and Transform Conditioning Image ---
            if use_mask:
                conditioning_path = mask_path
                dropout_prob = self.mask_dropout_prob
                log_msg = "mask"
                # Process as RGB mask
                def process_conditioning(conditioning_img):
                    conditioning_img = ensure_rgb(conditioning_img)
                    tensor = self.transform(conditioning_img)
                    return self.image_norm(tensor) # Normalize to [-1, 1]
            else: # use sketch
                conditioning_path = sketch_path
                dropout_prob = self.sketch_dropout_prob
                log_msg = "sketch"
                # Process as Grayscale sketch
                def process_conditioning(conditioning_img):
                    conditioning_img = ensure_grayscale(conditioning_img)
                    tensor = self.transform(conditioning_img) # [1, H, W] in [0, 1]
                    tensor = self.sketch_norm(tensor) # Normalize to [-1, 1]
                    return tensor.repeat(3, 1, 1) # [3, H, W]

            # --- Apply Modality Dropout ---
            if random.random() < dropout_prob:
                # Create a placeholder (all -1s after normalization)
                conditioning_tensor = torch.full((3, self.resolution, self.resolution), -1.0, dtype=torch.float32)
                logger_utils.debug(f"Dropped {log_msg} for sample {base_name} (Index {idx})")
            else:
                conditioning_img = Image.open(conditioning_path)
                conditioning_tensor = process_conditioning(conditioning_img)

            # --- Load and Process Text (with potential dropout) ---
            if random.random() < self.text_dropout_prob:
                chosen_caption = ""
                logger_utils.debug(f"Dropped text for sample {base_name} (Index {idx})")
            else:
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        captions = [line.strip() for line in f if line.strip()]
                    if not captions:
                        logger_utils.warning(f"Text file {text_path} is empty or contains only whitespace. Using empty caption.")
                        chosen_caption = ""
                    else:
                        chosen_caption = random.choice(captions)
                except UnicodeDecodeError:
                    logger_utils.error(f"Unicode decode error for text file {text_path}. Using empty caption.")
                    chosen_caption = ""
                except Exception as e:
                    logger_utils.error(f"Error reading text file {text_path}: {e}. Using empty caption.")
                    chosen_caption = ""

            # --- Tokenize Text with CLIP Tokenizer ---
            clip_tokenized_output = self.tokenizer(
                chosen_caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            clip_input_ids = clip_tokenized_output.input_ids.squeeze(0)

            # Return unified data structure
            return {
                "image": image_tensor,
                "conditioning_image": conditioning_tensor,
                "clip_input_ids": clip_input_ids,
                "modality": modality_flag # 0 for mask, 1 for sketch
            }
        except Exception as e:
            logger_utils.error(f"Error loading or processing data at index {idx} (Base Name: {base_name}): {e}")
            logger_utils.error(f"Image path: {img_path}")
            logger_utils.error(f"Mask path: {mask_path}")
            logger_utils.error(f"Sketch path: {sketch_path}")
            logger_utils.error(f"Text path: {text_path}")
            raise e