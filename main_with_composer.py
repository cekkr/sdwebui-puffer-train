import os
import glob
import json
import random
import shutil
import torch
import numpy as np
import datetime
import threading
import time
from PIL import Image, ImageOps
from modules import script_callbacks, shared, sd_models, ui
from modules.ui import create_refresh_button
from modules.processing import process_images
from modules.sd_models import load_model, reload_model_weights
import gradio as gr

# Import the compose_image function from local module
try:
    from images_composer import compose_image
except ImportError:
    # Define a fallback compose_image function in case the import fails
    def compose_image(foreground_path, background_paths, target_size, size_variation, max_ratio, captions=None):
        """
        Fallback function if images_composer module is not available
        
        Args:
            foreground_path (str): Path to the foreground image with transparent background
            background_paths (list): List of paths to background images
            target_size (tuple): Target size (width, height)
            size_variation (int): Variation in size allowed
            max_ratio (float): Maximum aspect ratio difference allowed
            captions (list, optional): List of available captions
            
        Returns:
            tuple: (PIL.Image, str) - Composed image and selected caption
        """
        # Load foreground with transparency
        try:
            foreground = Image.open(foreground_path).convert("RGBA")
            
            # Select random background
            bg_path = random.choice(background_paths)
            background = Image.open(bg_path).convert("RGB")
            
            # Calculate random size with variation
            width_var = random.randint(-size_variation, size_variation)
            height_var = random.randint(-size_variation, size_variation)
            width = target_size[0] + width_var
            height = target_size[1] + height_var
            
            # Ensure dimensions are divisible by 8 (common requirement for SD)
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            # Resize background
            background = background.resize((width, height), Image.LANCZOS)
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Skip if aspect ratio is too extreme
            if aspect_ratio > max_ratio or 1/aspect_ratio > max_ratio:
                print(f"Skipping composition due to extreme aspect ratio: {aspect_ratio}")
                return None, None
            
            # Resize foreground proportionally
            fg_aspect = foreground.width / foreground.height
            
            # Fit within the background
            new_fg_height = min(height, int(height * 0.9))  # Max 90% of background height
            new_fg_width = int(new_fg_height * fg_aspect)
            
            if new_fg_width > width * 0.9:  # If too wide, scale by width instead
                new_fg_width = int(width * 0.9)
                new_fg_height = int(new_fg_width / fg_aspect)
                
            foreground = foreground.resize((new_fg_width, new_fg_height), Image.LANCZOS)
            
            # Calculate position (center by default)
            pos_x = (width - new_fg_width) // 2
            pos_y = (height - new_fg_height) // 2
            
            # Add some randomness to position
            pos_x += random.randint(-20, 20)
            pos_y += random.randint(-20, 20)
            
            # Ensure it stays within bounds
            pos_x = max(0, min(pos_x, width - new_fg_width))
            pos_y = max(0, min(pos_y, height - new_fg_height))
            
            # Paste foreground onto background
            background.paste(foreground, (pos_x, pos_y), foreground)
            
            # Select or create caption
            caption = ""
            if captions and len(captions) > 0:
                caption = random.choice(captions)
            else:
                # Extract caption from foreground filename if no captions provided
                base_name = os.path.basename(foreground_path)
                caption = os.path.splitext(base_name)[0].replace("_", " ")
            
            return background, caption
            
        except Exception as e:
            print(f"Error in compose_image fallback: {e}")
            return None, None

# Constants for file paths
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_PATH = os.path.join(EXTENSION_PATH, "training_data")
SAVED_STATES_PATH = os.path.join(EXTENSION_PATH, "saved_states")
BACKGROUNDS_PATH = os.path.join(EXTENSION_PATH, "backgrounds")
TRANSPARENT_DATASET_PATH = os.path.join(EXTENSION_PATH, "transparent_dataset")
CONFIG_PATH = os.path.join(EXTENSION_PATH, "config.json")
COMPOSED_CACHE_PATH = os.path.join(EXTENSION_PATH, "composed_cache")

# Ensure necessary directories exist
for path in [TRAINING_DATA_PATH, SAVED_STATES_PATH, BACKGROUNDS_PATH, 
             TRANSPARENT_DATASET_PATH, COMPOSED_CACHE_PATH]:
    os.makedirs(path, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "learning_rate": 1e-6,
    "train_batch_size": 1,
    "max_train_steps": 1000,
    "gradient_accumulation_steps": 4,
    "flip_prob": 0.5,
    "use_backgrounds": True,  # Changed to True as per new functionality
    "transparent_images": False,  # Disabled as it's resource-intensive
    "save_every_n_steps": 100,
    "validation_prompt": "A photo of person",
    "last_saved_step": 0,
    "total_trained_steps": 0,
    "dynamic_resolution": True,  # Changed to True for dynamic training
    "use_bucketing": True,  # Changed to True for better training
    "smart_resizing": True,
    "target_width": 512,
    "target_height": 512,
    "background_color": [255, 255, 255, 255],
    "bucket_min_width": 384,
    "bucket_max_width": 768,
    "bucket_min_height": 384, 
    "bucket_max_height": 768,
    "bucket_step": 64,
    # New configuration options for dynamic composition
    "compose_frequency": 128,  # Regenerate composed images every N steps
    "compose_batch_size": 64,  # Number of images to compose in one batch
    "target_size_variation": 64,  # Variation in size during composition
    "max_aspect_ratio": 1.5,  # Maximum aspect ratio allowed
    "use_image_composer": True,  # Whether to use the image composer
    "random_caption_mixing": True,  # Whether to randomly mix captions
}

# Load or create configuration
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Ensure new config keys are added
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    else:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

config = load_config()

def save_config():
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

class CompositionThread(threading.Thread):
    """Thread class to compose images in the background during training"""
    
    def __init__(self, trainer, batch_size=64):
        threading.Thread.__init__(self)
        self.trainer = trainer
        self.batch_size = batch_size
        self.daemon = True  # Daemon thread terminates when main program exits
        self.stop_event = threading.Event()
        self.composed_images = []
        self.lock = threading.Lock()
    
    def run(self):
        """Main thread function that composes images"""
        while not self.stop_event.is_set():
            # Check if we need to generate more images
            with self.lock:
                need_more = len(self.composed_images) < self.batch_size
            
            if need_more and self.trainer.foreground_images and self.trainer.backgrounds:
                try:
                    # Compose a new image
                    foreground = random.choice(self.trainer.foreground_images)
                    
                    # Get caption sources
                    base_path = os.path.splitext(foreground)[0]
                    txt_path = base_path + ".txt"
                    captions = []
                    
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            captions = [line.strip() for line in f.readlines() if line.strip()]
                    
                    # Set target size with variation
                    target_width = config["target_width"]
                    target_height = config["target_height"]
                    variation = config["target_size_variation"]
                    max_ratio = config["max_aspect_ratio"]
                    
                    # Call the compose_image function
                    img, caption = compose_image(
                        foreground_path=foreground,
                        background_paths=self.trainer.backgrounds,
                        target_size=(target_width, target_height),
                        size_variation=variation,
                        max_ratio=max_ratio,
                        captions=captions
                    )
                    
                    if img is not None and caption is not None:
                        # Save to cache for potential reuse
                        timestamp = int(time.time() * 1000)
                        cache_path = os.path.join(COMPOSED_CACHE_PATH, f"composed_{timestamp}.jpg")
                        caption_path = os.path.join(COMPOSED_CACHE_PATH, f"composed_{timestamp}.txt")
                        
                        img.save(cache_path)
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                        
                        with self.lock:
                            self.composed_images.append({
                                "image_path": cache_path,
                                "caption": caption
                            })
                except Exception as e:
                    print(f"Error in composition thread: {e}")
            
            # Sleep to avoid CPU overuse
            time.sleep(0.1)
    
    def get_composed_image(self):
        """Get a composed image from the thread's cache"""
        with self.lock:
            if self.composed_images:
                return self.composed_images.pop(0)
            return None
    
    def stop(self):
        """Stop the thread"""
        self.stop_event.set()

class AdvancedTrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_step = 0
        self.total_steps = 0
        self.training_active = False
        self.checkpoint_path = None
        self.dataset = []
        self.backgrounds = []
        self.foreground_images = []
        self.composition_thread = None
        self.next_compose_step = 0
    
    def start_composition_thread(self):
        """Start the background thread for image composition"""
        if self.composition_thread is None or not self.composition_thread.is_alive():
            self.composition_thread = CompositionThread(self, config["compose_batch_size"])
            self.composition_thread.start()
            print("Started background image composition thread")
    
    def stop_composition_thread(self):
        """Stop the background thread for image composition"""
        if self.composition_thread and self.composition_thread.is_alive():
            self.composition_thread.stop()
            self.composition_thread.join(timeout=2.0)
            print("Stopped background image composition thread")
            self.composition_thread = None

    def load_model_for_training(self, model_name):
        """Load the model and prepare it for training"""
        print(f"Loading model {model_name} for training...")
        self.model = load_model(model_name)
        self.model.train()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"]
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["max_train_steps"]
        )
        
        print("Model loaded and ready for training")
        return True

    def prepare_dataset(self, dataset_path):
        """Prepare the dataset with support for transparent images and image composition"""
        self.dataset = []
        
        # Get all image files
        image_files = glob.glob(os.path.join(dataset_path, "*.png")) + \
                      glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                      glob.glob(os.path.join(dataset_path, "*.jpeg"))
        
        # Keep track of foreground transparent images for composition
        if config["use_image_composer"]:
            self.foreground_images = glob.glob(os.path.join(TRANSPARENT_DATASET_PATH, "*.png"))
            print(f"Found {len(self.foreground_images)} transparent foreground images")
        
        # Prepare aspect ratio buckets if enabled
        if config["use_bucketing"]:
            self._prepare_buckets()
        
        # Collect information about all images for potential bucketing
        image_dimensions = []
        
        for img_path in image_files:
            base_path = os.path.splitext(img_path)[0]
            txt_path = base_path + ".txt"
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    captions = [line.strip() for line in f.readlines() if line.strip()]
                
                # Get image dimensions if we're using dynamic resolution or bucketing
                if config["dynamic_resolution"] or config["use_bucketing"]:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            # Make dimensions divisible by 8
                            width = (width // 8) * 8
                            height = (height // 8) * 8
                            aspect_ratio = width / height
                            
                            image_dimensions.append({
                                "path": img_path,
                                "width": width,
                                "height": height,
                                "ratio": aspect_ratio
                            })
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue
                
                self.dataset.append({
                    "image_path": img_path,
                    "captions": captions
                })
        
        # Load backgrounds for composition
        self.backgrounds = glob.glob(os.path.join(BACKGROUNDS_PATH, "*.png")) + \
                          glob.glob(os.path.join(BACKGROUNDS_PATH, "*.jpg")) + \
                          glob.glob(os.path.join(BACKGROUNDS_PATH, "*.jpeg"))
        
        # Start the composition thread if required
        if config["use_image_composer"] and self.foreground_images and self.backgrounds:
            self.start_composition_thread()
            
            # Schedule first composition
            self.next_compose_step = config["compose_frequency"]
        
        # Display statistics about the dataset
        print(f"Dataset prepared with {len(self.dataset)} images")
        print(f"Loaded {len(self.backgrounds)} backgrounds")
        
        if config["dynamic_resolution"] and image_dimensions:
            # Calculate statistics about image dimensions
            widths = [img["width"] for img in image_dimensions]
            heights = [img["height"] for img in image_dimensions]
            ratios = [img["ratio"] for img in image_dimensions]
            
            print(f"Image dimension statistics:")
            print(f"  Width range: {min(widths)} to {max(widths)}")
            print(f"  Height range: {min(heights)} to {max(heights)}")
            print(f"  Aspect ratio range: {min(ratios):.2f} to {max(ratios):.2f}")
            
            if config["use_bucketing"]:
                print(f"Using {len(self.buckets)} aspect ratio buckets")
                for i, bucket in enumerate(self.buckets):
                    print(f"  Bucket {i+1}: {bucket['width']}x{bucket['height']} (ratio: {bucket['ratio']:.2f})")
        
        return len(self.dataset)
        
    def _prepare_buckets(self):
        """Prepare aspect ratio buckets for training"""
        self.buckets = []
        
        # Create buckets with different aspect ratios
        min_width = config["bucket_min_width"]
        max_width = config["bucket_max_width"]
        min_height = config["bucket_min_height"]
        max_height = config["bucket_max_height"]
        step = config["bucket_step"]
        
        # Generate buckets covering common aspect ratios
        # Portrait buckets (tall)
        for width in range(min_width, max_width + 1, step):
            for height in range(width, max_height + 1, step):
                if height > max_height:
                    continue
                # Make sure dimensions are multiples of 8
                width_adjusted = (width // 8) * 8
                height_adjusted = (height // 8) * 8
                if width_adjusted == 0 or height_adjusted == 0:
                    continue
                ratio = width_adjusted / height_adjusted
                self.buckets.append({
                    "width": width_adjusted,
                    "height": height_adjusted,
                    "ratio": ratio,
                    "pixels": width_adjusted * height_adjusted
                })
        
        # Landscape buckets (wide)
        for height in range(min_height, max_height + 1, step):
            for width in range(height, max_width + 1, step):
                if width > max_width:
                    continue
                # Make sure dimensions are multiples of 8
                width_adjusted = (width // 8) * 8
                height_adjusted = (height // 8) * 8
                if width_adjusted == 0 or height_adjusted == 0:
                    continue
                ratio = width_adjusted / height_adjusted
                # Check if this bucket already exists (avoid duplicates)
                if not any(b["width"] == width_adjusted and b["height"] == height_adjusted for b in self.buckets):
                    self.buckets.append({
                        "width": width_adjusted,
                        "height": height_adjusted,
                        "ratio": ratio,
                        "pixels": width_adjusted * height_adjusted
                    })
        
        # Sort buckets by resolution (total pixels)
        self.buckets.sort(key=lambda x: x["pixels"])
        
        # Limit to a reasonable number of buckets if there are too many
        max_buckets = 20
        if len(self.buckets) > max_buckets:
            # Keep buckets distributed across the ratio range
            ratios = [b["ratio"] for b in self.buckets]
            min_ratio, max_ratio = min(ratios), max(ratios)
            ratio_range = max_ratio - min_ratio
            
            step = len(self.buckets) / max_buckets
            indices = [int(i * step) for i in range(max_buckets)]
            self.buckets = [self.buckets[i] for i in indices]

    def preprocess_image(self, image_path, caption):
        """Preprocess image with dynamic sizing and horizontal flipping"""
        # We're now ensuring all images are RGB (no transparency)
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        
        # Random horizontal flip
        if random.random() < config["flip_prob"]:
            img = ImageOps.mirror(img)
        
        # Handle image dimensions based on configuration
        if config["dynamic_resolution"]:
            # If dynamic resolution is enabled, use the original image dimensions
            # But make sure they're divisible by 8 (required by most SD models)
            target_width = (original_size[0] // 8) * 8
            target_height = (original_size[1] // 8) * 8
            
            if target_width != original_size[0] or target_height != original_size[1]:
                img = img.resize((target_width, target_height), Image.LANCZOS)
            
            # Calculate aspect ratio for logging/metadata
            aspect_ratio = target_width / target_height
            
            # If using bucketing, find the closest bucket for this aspect ratio
            if config["use_bucketing"] and hasattr(self, 'buckets'):
                closest_bucket = min(self.buckets, key=lambda b: abs(b['ratio'] - aspect_ratio))
                target_width, target_height = closest_bucket['width'], closest_bucket['height']
                img = img.resize((target_width, target_height), Image.LANCZOS)
        else:
            # Fixed dimensions specified in config
            target_size = (config["target_width"], config["target_height"])
            img = img.resize(target_size, Image.LANCZOS)
        
        return img, caption

    def save_training_state(self, save_name):
        """Save the current training state for later resuming"""
        if not self.model:
            return False
        
        state_path = os.path.join(SAVED_STATES_PATH, f"{save_name}.pt")
        
        # Save the complete training state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'config': config
        }, state_path)
        
        # Also save a usable checkpoint
        checkpoint_path = os.path.join(SAVED_STATES_PATH, f"{save_name}_checkpoint.ckpt")
        self.export_checkpoint(checkpoint_path)
        
        print(f"Training state saved to {state_path}")
        print(f"Usable checkpoint saved to {checkpoint_path}")
        
        config["last_saved_step"] = self.current_step
        config["total_trained_steps"] = self.total_steps
        save_config()
        
        return True

    def resume_training(self, state_name):
        """Resume training from a saved state"""
        state_path = os.path.join(SAVED_STATES_PATH, f"{state_name}.pt")
        
        if not os.path.exists(state_path):
            print(f"State file {state_path} not found")
            return False
        
        # Load the saved state
        checkpoint = torch.load(state_path)
        
        # Load model if not already loaded
        if not self.model:
            self.load_model_for_training("some_default_model")  # This would need to be handled properly
        
        # Restore the training state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['current_step']
        self.total_steps = checkpoint['total_steps']
        
        # Restore config settings
        global config
        config = checkpoint['config']
        save_config()
        
        print(f"Training resumed from step {self.current_step}")
        return True

    def export_checkpoint(self, output_path):
        """Export a usable checkpoint"""
        if not self.model:
            return False
        
        # Convert the model to a format that can be loaded by Stable Diffusion WebUI
        # This depends on the specific format expected by the WebUI
        
        # Create a dictionary with the model weights and metadata
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "epoch": self.current_step,
            "global_step": self.total_steps,
            "iteration": self.current_step,
            "model_name": "Advanced Training Checkpoint",
            "sd_checkpoint": config.get("base_model_name", ""),
            "sd_checkpoint_name": config.get("base_model_name", ""),
            "training_params": {
                "learning_rate": config["learning_rate"],
                "batch_size": config["train_batch_size"],
                "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                "resolution": f"{config['target_width']}x{config['target_height']}",
                "dynamic_resolution": config["dynamic_resolution"],
                "use_bucketing": config["use_bucketing"],
                "use_image_composer": config["use_image_composer"],
                "dataset_size": len(self.dataset) if hasattr(self, 'dataset') else 0,
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save the checkpoint
        try:
            torch.save(checkpoint, output_path)
            
            # Create a small preview image for the checkpoint if we have validation prompt
            if config["validation_prompt"]:
                try:
                    # Try to generate a preview image using the current model
                    preview_prompt = config["validation_prompt"]
                    preview_path = os.path.splitext(output_path)[0] + "_preview.png"
                    
                    # This would require integration with the WebUI processing pipeline
                    # For demonstration, we'll use a placeholder
                    # In a full implementation, you'd use:
                    # preview_image = process_images(model=self.model, prompt=preview_prompt)
                    # preview_image.save(preview_path)
                    
                    print(f"Preview would be generated with prompt: '{preview_prompt}'")
                except Exception as e:
                    print(f"Failed to generate preview: {e}")
            
            # Create a metadata file with information about the checkpoint
            metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
            with open(metadata_path, 'w') as f:
                # Extract non-tensor information for the metadata file
                metadata = {
                    "model_name": "Advanced Training Checkpoint",
                    "created_at": datetime.datetime.now().isoformat(),
                    "steps": self.current_step,
                    "total_steps": self.total_steps,
                    "base_model": config.get("base_model_name", ""),
                    "training_config": {k: v for k, v in config.items() if not isinstance(v, torch.Tensor)}
                }
                json.dump(metadata, f, indent=2)
            
            print(f"Checkpoint exported to {output_path}")
            print(f"Metadata saved to {metadata_path}")
            return True
        except Exception as e:
            print(f"Error exporting checkpoint: {e}")
            return False

    def train_step(self):
        """Perform a single training step with dynamic image composition"""
        if not self.model or not self.dataset:
            return {"error": "Model or dataset not loaded"}
        
        # Check if we need to regenerate composed images
        if config["use_image_composer"] and self.current_step >= self.next_compose_step:
            # Schedule next composition
            self.next_compose_step = self.current_step + config["compose_frequency"]
            print(f"Time to compose new images. Next composition at step {self.next_compose_step}")
            
            # Make sure composition thread is running
            if not self.composition_thread or not self.composition_thread.is_alive():
                self.start_composition_thread()
        
        # Try to use a composed image if available
        composed_sample = None
        if config["use_image_composer"] and self.composition_thread:
            composed_sample = self.composition_thread.get_composed_image()
        
        if composed_sample:
            # Use the composed image and caption
            img_path = composed_sample["image_path"]
            caption = composed_sample["caption"]
            img = Image.open(img_path).convert("RGB")  # Ensure RGB format
            
            # No need for preprocessing as the composed image is already processed
            img_tensor = img  # In real implementation, convert to tensor
        else:
            # Fallback to standard dataset
            sample = random.choice(self.dataset)
            # Select a random caption for this image
            caption = random.choice(sample["captions"])
            
            # Preprocess the image and caption
            img, caption = self.preprocess_image(sample["image_path"], caption)
            img_tensor = img  # In real implementation, convert to tensor
        
        # Perform training step (this is a simplified version)
        self.optimizer.zero_grad()
        
        # Here you would:
        # 1. Convert image to latent representation
        # 2. Encode the caption
        # 3. Forward pass
        # 4. Calculate loss
        # 5. Backward pass
        # 6. Update weights
        
        # For this example, we'll just increment the step counters
        self.current_step += 1
        self.total_steps += 1
        
        # Update the scheduler
        self.scheduler.step()
        
        # Save training state if needed
        if self.current_step % config["save_every_n_steps"] == 0:
            self.save_training_state(f"auto_save_step_{self.current_step}")
        
        return {
            "step": self.current_step,
            "total_steps": self.total_steps,
            "image": img,
            "caption": caption,
            "loss": 0.0  # Placeholder
        }

    def start_training(self, max_steps):
        """Start or continue the training process"""
        if not self.model or not self.dataset:
            return False
        
        self.training_active = True
        target_step = self.current_step + max_steps
        
        # Make sure composition thread is running if needed
        if config["use_image_composer"] and self.foreground_images and self.backgrounds:
            self.start_composition_thread()
        
        print(f"Training started from step {self.current_step} to {target_step}")
        while self.current_step < target_step and self.training_active:
            step_result = self.train_step()
            
            # Here you would update UI, log progress, etc.
            if self.current_step % 10 == 0:
                print(f"Step {self.current_step}/{target_step}, Loss: {step_result['loss']}")
        
        if self.current_step >= target_step:
            print("Training completed")
        else:
            print("Training paused")
        
        self.training_active = False
        return True

    def pause_training(self):
        """Pause the training process"""
        self.training_active = False
        print("Training will pause after the current step")
        
        # Stop composition thread when pausing
        self.stop_composition_thread()
        return True
    


# Create the singleton instance
trainer = AdvancedTrainer()

# UI Components
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as advanced_training_tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Advanced Training Extension")
                gr.Markdown("Train Stable Diffusion models with advanced features like pause/resume, transparent datasets, and background variation.")
        
        with gr.Tabs():
            with gr.TabItem("Model Setup"):
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            label="Model", 
                            choices=sd_models.checkpoint_tiles(), 
                            elem_id="model_dropdown"
                        )
                        refresh_button = create_refresh_button(model_dropdown, sd_models.list_models, "refresh_model_dropdown")
                        load_model_button = gr.Button("Load Model for Training")
                        model_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.TabItem("Dataset Preparation"):
                with gr.Row():
                    with gr.Column():
                        dataset_path = gr.Textbox(label="Dataset Path", value=TRAINING_DATA_PATH)
                        transparent_dataset = gr.Checkbox(label="Use Transparent Dataset", value=config["transparent_images"])
                        use_backgrounds = gr.Checkbox(label="Use Random Backgrounds", value=config["use_backgrounds"])
                        backgrounds_path = gr.Textbox(label="Backgrounds Path", value=BACKGROUNDS_PATH)
                        process_dataset_button = gr.Button("Process Dataset")
                        dataset_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Image Resolution Options")
                        dynamic_resolution = gr.Checkbox(label="Use Dynamic Resolution", value=config["dynamic_resolution"], 
                                                     info="Train with native image resolution (aspect ratio preserved)")
                        use_bucketing = gr.Checkbox(label="Use Aspect Ratio Bucketing", value=config["use_bucketing"], 
                                                 info="Group similar aspect ratios together for more efficient training")
                        smart_resizing = gr.Checkbox(label="Smart Resizing for Transparent Images", value=config["smart_resizing"], 
                                                  info="Automatically resize based on non-transparent content")
                    
                    with gr.Column():
                        gr.Markdown("### Fixed Resolution Settings")
                        target_width = gr.Slider(label="Target Width", minimum=256, maximum=1024, value=config["target_width"], step=8)
                        target_height = gr.Slider(label="Target Height", minimum=256, maximum=1024, value=config["target_height"], step=8)
                        background_color = gr.ColorPicker(label="Background Color (for transparent images)", value="#FFFFFF")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Advanced Bucketing Settings")
                        bucket_min_width = gr.Slider(label="Min Bucket Width", minimum=256, maximum=1024, value=config["bucket_min_width"], step=64)
                        bucket_max_width = gr.Slider(label="Max Bucket Width", minimum=256, maximum=1024, value=config["bucket_max_width"], step=64)
                        bucket_min_height = gr.Slider(label="Min Bucket Height", minimum=256, maximum=1024, value=config["bucket_min_height"], step=64)
                        bucket_max_height = gr.Slider(label="Max Bucket Height", minimum=256, maximum=1024, value=config["bucket_max_height"], step=64)
                        bucket_step = gr.Slider(label="Bucket Size Step", minimum=32, maximum=128, value=config["bucket_step"], step=32)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Create Transparent Dataset")
                        input_dataset_path = gr.Textbox(label="Input Dataset Path")
                        output_transparent_path = gr.Textbox(label="Output Transparent Path", value=TRANSPARENT_DATASET_PATH)
                        create_transparent_button = gr.Button("Create Transparent Dataset")
                        transparent_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.TabItem("Training Configuration"):
                with gr.Row():
                    with gr.Column():
                        learning_rate = gr.Slider(label="Learning Rate", minimum=1e-7, maximum=1e-5, value=config["learning_rate"], step=1e-7)
                        batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=config["train_batch_size"], step=1)
                        max_steps = gr.Slider(label="Max Training Steps", minimum=100, maximum=10000, value=config["max_train_steps"], step=100)
                        gradient_accumulation = gr.Slider(label="Gradient Accumulation Steps", minimum=1, maximum=8, value=config["gradient_accumulation_steps"], step=1)
                        flip_probability = gr.Slider(label="Horizontal Flip Probability", minimum=0.0, maximum=1.0, value=config["flip_prob"], step=0.1)
                        save_steps = gr.Slider(label="Save Every N Steps", minimum=50, maximum=1000, value=config["save_every_n_steps"], step=50)
                    
                    with gr.Column():
                        validation_prompt = gr.Textbox(label="Validation Prompt", value=config["validation_prompt"])
                        save_config_button = gr.Button("Save Configuration")
                        config_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.TabItem("Training Control"):
                with gr.Row():
                    with gr.Column():
                        start_button = gr.Button("Start Training")
                        pause_button = gr.Button("Pause Training")
                        training_steps = gr.Slider(label="Steps to Train", minimum=10, maximum=1000, value=100, step=10)
                    
                    with gr.Column():
                        current_step = gr.Textbox(label="Current Step", value="0", interactive=False)
                        total_trained = gr.Textbox(label="Total Steps Trained", value="0", interactive=False)
                        training_status = gr.Textbox(label="Training Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        save_state_name = gr.Textbox(label="Save State Name", value="my_training_state")
                        save_state_button = gr.Button("Save Training State")
                        save_status = gr.Textbox(label="Save Status", interactive=False)
                    
                    with gr.Column():
                        resume_state_dropdown = gr.Dropdown(label="Resume from State", choices=[], elem_id="resume_state_dropdown")
                        refresh_states_button = gr.Button("Refresh States")
                        resume_button = gr.Button("Resume Training")
                        resume_status = gr.Textbox(label="Resume Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        export_checkpoint_name = gr.Textbox(label="Export Checkpoint Name", value="my_checkpoint")
                        export_button = gr.Button("Export Checkpoint")
                        export_status = gr.Textbox(label="Export Status", interactive=False)
        
        # Event handlers
        def refresh_states():
            state_files = glob.glob(os.path.join(SAVED_STATES_PATH, "*.pt"))
            return [os.path.basename(f)[:-3] for f in state_files]
        
        refresh_states_button.click(fn=refresh_states, outputs=[resume_state_dropdown])
        
        def load_model_for_training(model_name):
            success = trainer.load_model_for_training(model_name)
            return "Model loaded successfully" if success else "Failed to load model"
        
        load_model_button.click(fn=load_model_for_training, inputs=[model_dropdown], outputs=[model_status])
        
        def process_dataset(path, use_transparent, use_bg, bg_path):
            config["transparent_images"] = use_transparent
            config["use_backgrounds"] = use_bg
            save_config()
            
            count = trainer.prepare_dataset(path)
            return f"Dataset processed: {count} images found"
        
        process_dataset_button.click(fn=process_dataset, inputs=[dataset_path, transparent_dataset, use_backgrounds, backgrounds_path], outputs=[dataset_status])
        
        def create_transparent_dataset(input_path, output_path):
            count = trainer.process_transparent_dataset(input_path, output_path)
            return f"Transparent dataset created with {count} images"
        
        create_transparent_button.click(fn=create_transparent_dataset, inputs=[input_dataset_path, output_transparent_path], outputs=[transparent_status])
        
        def save_training_configuration(lr, bs, ms, ga, fp, ss, vp):
            config["learning_rate"] = lr
            config["train_batch_size"] = bs
            config["max_train_steps"] = ms
            config["gradient_accumulation_steps"] = ga
            config["flip_prob"] = fp
            config["save_every_n_steps"] = ss
            config["validation_prompt"] = vp
            save_config()
            return "Configuration saved"
        
        save_config_button.click(fn=save_training_configuration, 
                              inputs=[learning_rate, batch_size, max_steps, gradient_accumulation, flip_probability, save_steps, validation_prompt], 
                              outputs=[config_status])
                              
        def save_dataset_configuration(dynamic, bucketing, smart, t_width, t_height, bg_color, min_w, max_w, min_h, max_h, step):
            config["dynamic_resolution"] = dynamic
            config["use_bucketing"] = bucketing
            config["smart_resizing"] = smart
            config["target_width"] = t_width
            config["target_height"] = t_height
            # Convert color picker hex to RGBA
            if bg_color.startswith('#'):
                r = int(bg_color[1:3], 16)
                g = int(bg_color[3:5], 16)
                b = int(bg_color[5:7], 16)
                config["background_color"] = [r, g, b, 255]
            config["bucket_min_width"] = min_w
            config["bucket_max_width"] = max_w
            config["bucket_min_height"] = min_h
            config["bucket_max_height"] = max_h
            config["bucket_step"] = step
            save_config()
            return "Dataset configuration saved"
            
        # Connect dataset configuration UI elements
        save_dataset_button = gr.Button("Save Resolution Settings")
        save_dataset_button.click(
            fn=save_dataset_configuration,
            inputs=[
                dynamic_resolution, use_bucketing, smart_resizing,
                target_width, target_height, background_color,
                bucket_min_width, bucket_max_width, bucket_min_height, bucket_max_height, bucket_step
            ],
            outputs=[dataset_status]
        )
        
        def start_training(steps):
            success = trainer.start_training(int(steps))
            return ("Training started" if success else "Failed to start training", 
                    str(trainer.current_step), 
                    str(trainer.total_steps))
        
        start_button.click(fn=start_training, inputs=[training_steps], outputs=[training_status, current_step, total_trained])
        
        def pause_training():
            success = trainer.pause_training()
            return "Training will pause after current step" if success else "No active training to pause"
        
        pause_button.click(fn=pause_training, outputs=[training_status])
        
        def save_state(name):
            success = trainer.save_training_state(name)
            new_states = refresh_states()
            return "Training state saved successfully" if success else "Failed to save state", new_states
        
        save_state_button.click(fn=save_state, inputs=[save_state_name], outputs=[save_status, resume_state_dropdown])
        
        def resume_training(state_name):
            success = trainer.resume_training(state_name)
            return ("Training state loaded successfully" if success else "Failed to load training state",
                    str(trainer.current_step),
                    str(trainer.total_steps))
        
        resume_button.click(fn=resume_training, inputs=[resume_state_dropdown], outputs=[resume_status, current_step, total_trained])
        
        def export_checkpoint(name):
            path = os.path.join(SAVED_STATES_PATH, f"{name}.ckpt")
            success = trainer.export_checkpoint(path)
            return f"Checkpoint exported to {path}" if success else "Failed to export checkpoint"
        
        export_button.click(fn=export_checkpoint, inputs=[export_checkpoint_name], outputs=[export_status])
        
        # Initialize UI
        refresh_states_button.click(fn=refresh_states, outputs=[resume_state_dropdown])
    
    return [(advanced_training_tab, "Advanced Training", "advanced_training_tab")]

# Register the extension
script_callbacks.on_ui_tabs(on_ui_tabs)

# Function to preprocess a folder of images to create transparent versions
def preprocess_transparent_dataset(input_folder, output_folder):
    """
    Process all images in the input folder to remove backgrounds and save them
    as transparent PNGs in the output folder.
    
    This would require a background removal model. For a real implementation,
    you could use something like rembg or u2net.
    """
    # Placeholder for actual implementation
    pass

def setup_extension():
    """Initial setup when the extension is loaded"""
    global config
    config = load_config()
    
    # Create necessary directories
    for path in [TRAINING_DATA_PATH, SAVED_STATES_PATH, BACKGROUNDS_PATH, TRANSPARENT_DATASET_PATH]:
        os.makedirs(path, exist_ok=True)
    
    print("Advanced Training Extension initialized")

# Run setup on import
setup_extension()