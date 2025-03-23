import os
from PIL import Image, ImageDraw
from mtcnn import MTCNN
import piexif
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import argparse
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from functools import partial

# Comprehensive TensorFlow warning suppression
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.keras.utils.disable_interactive_logging()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.set_soft_device_placement(True)

# Global face detector for the main process
face_detector = None

def init_worker():
    """Initialize worker process with its own MTCNN instance."""
    global face_detector
    face_detector = MTCNN()

def rotate_image(image):
    """Rotate image based on EXIF orientation using PIL's built-in handling."""
    try:
        # Get EXIF data
        exif = image._getexif()
        if exif is None:
            return image
            
        # Get orientation from EXIF
        orientation_key = 274  # 274 is the EXIF key for orientation
        if orientation_key in exif:
            orientation = exif[orientation_key]
            
            # Rotate image based on orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
                # Set new orientation to 1 (normal)
                exif[orientation_key] = 1
            elif orientation == 6:
                image = image.rotate(270, expand=True)
                # Set new orientation to 1 (normal)
                exif[orientation_key] = 1
            elif orientation == 8:
                image = image.rotate(90, expand=True)
                # Set new orientation to 1 (normal)
                exif[orientation_key] = 1
                
            # Convert EXIF dict to bytes
            exif_bytes = piexif.dump(exif)
            image._exif = exif_bytes
            
    except Exception as e:
        logging.warning(f"Could not process EXIF orientation: {str(e)}")
    
    return image

def get_face_center(image, show_faces=False):
    """Detect faces and return the center point of all faces using MTCNN."""
    # Store original size
    original_width, original_height = image.size
    
    # Calculate resize factor to make the image smaller but maintain aspect ratio
    MAX_DIMENSION = 800  # Maximum dimension for face detection
    scale = min(MAX_DIMENSION / original_width, MAX_DIMENSION / original_height)
    
    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image for faster face detection
    small_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_array = np.array(small_image)
    
    # Detect faces on smaller image
    faces = face_detector.detect_faces(img_array)
    
    # Filter faces by confidence threshold
    CONFIDENCE_THRESHOLD = 0.95  # Increased threshold to reduce false positives
    faces = [face for face in faces if face['confidence'] >= CONFIDENCE_THRESHOLD]
    
    if not faces:
        return None
    
    # If show_faces mode, draw bounding boxes on original image
    if show_faces:
        draw = ImageDraw.Draw(image)
        for face in faces:
            # Scale coordinates back to original size
            x, y, width, height = face['box']
            x = int(x / scale)
            y = int(y / scale)
            width = int(width / scale)
            height = int(height / scale)
            confidence = face['confidence']
            # Draw rectangle with red outline and confidence score
            draw.rectangle([x, y, x + width, y + height], outline='red', width=2)
            # Add confidence score
            draw.text((x, y - 10), f"{confidence:.2f}", fill='red')
    
    # Calculate the center point of all faces (scaled back to original size)
    total_x = 0
    total_y = 0
    for face in faces:
        x, y, width, height = face['box']
        # Scale coordinates back to original size
        x = int(x / scale)
        y = int(y / scale)
        width = int(width / scale)
        height = int(height / scale)
        x_center = x + width / 2
        y_center = y + height / 2
        total_x += x_center
        total_y += y_center
    
    return total_x / len(faces), total_y / len(faces)

def make_square(image, face_center=None, ratio=None):
    """Crop image to specified ratio, centering on faces if detected."""
    width, height = image.size
    
    if ratio:
        # Parse ratio string (e.g., "16:9")
        try:
            target_width, target_height = map(int, ratio.split(':'))
            # Calculate crop dimensions based on ratio
            if width/height > target_width/target_height:
                # Image is wider than target ratio
                crop_width = int(height * (target_width/target_height))
                crop_height = height
            else:
                # Image is taller than target ratio
                crop_width = width
                crop_height = int(width * (target_height/target_width))
        except (ValueError, ZeroDivisionError):
            logging.error(f"Invalid ratio format: {ratio}. Using square crop.")
            crop_width = crop_height = min(width, height)
    else:
        # Square crop
        crop_width = crop_height = min(width, height)
    
    if face_center:
        # Calculate crop box centered on faces
        face_x, face_y = face_center
        left = max(0, face_x - crop_width/2)
        top = max(0, face_y - crop_height/2)
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)
        
        # Adjust if we hit the edges
        if right - left < crop_width:
            left = right - crop_width
        if bottom - top < crop_height:
            top = bottom - crop_height
    else:
        # Center crop if no faces detected
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        right = left + crop_width
        bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))

def process_single_image(args):
    """Process a single image with its arguments."""
    input_path, output_path, show_faces, ratio, overwrite = args
    try:
        # Skip if output file exists and overwrite is False
        if not overwrite and os.path.exists(output_path):
            return
            
        # Open image
        image = Image.open(input_path)
        
        # Rotate image based on EXIF
        image = rotate_image(image)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect faces and get center point
        face_center = get_face_center(image, show_faces)
        
        # Make square or custom ratio
        cropped_image = make_square(image, face_center, ratio)
        
        # Save the processed image with EXIF data only if it exists
        if hasattr(image, '_exif') and image._exif is not None:
            cropped_image.save(output_path, quality=95, exif=image._exif)
        else:
            cropped_image.save(output_path, quality=95)
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images to make them square and face-centered.')
    parser.add_argument('--show-faces', action='store_true', help='Show face detection boxes and confidence scores on output images')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--input-dir', type=str, default='photos', help='Input directory containing images (default: photos)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for processed images (default: output)')
    parser.add_argument('--ratio', type=str, default=None, help='Crop ratio in format width:height (e.g., 16:9, 4:3). Default is square.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files in output directory')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get list of images to process
    photos_dir = Path(args.input_dir)
    if not photos_dir.exists():
        print(f"❌ Input directory '{args.input_dir}' does not exist!")
        return
        
    image_files = list(photos_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print(f"❌ No images found in the input directory '{args.input_dir}'!")
        return
    
    # Count existing files
    existing_files = [f for f in image_files if (output_dir / f.name).exists()]
    if existing_files and not args.overwrite:
        print(f"📝 Found {len(existing_files)} existing files in output directory")
        print("💡 Use --overwrite to process these files")
        image_files = [f for f in image_files if not (output_dir / f.name).exists()]
    
    if not image_files:
        print("✨ No new images to process!")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    if args.show_faces:
        print("👤 Face detection boxes and confidence scores will be shown on output images")
    if args.ratio:
        print(f"📐 Crop ratio set to {args.ratio}")
    if args.overwrite:
        print("🔄 Overwrite mode enabled - existing files will be processed")
    
    # Prepare arguments for parallel processing
    process_args = [(str(f), str(output_dir / f.name), args.show_faces, args.ratio, args.overwrite) for f in image_files]
    
    # Determine number of worker processes
    num_workers = args.workers if args.workers is not None else cpu_count()
    print(f"⚡ Using {num_workers} worker processes")
    
    # Process images in parallel with progress bar
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        list(tqdm(
            pool.imap(process_single_image, process_args),
            total=len(process_args),
            desc="Processing images",
            unit="image"
        ))
    
    print(f"\n✅ Processing complete! Check the '{args.output_dir}' directory for results.")

if __name__ == "__main__":
    main() 