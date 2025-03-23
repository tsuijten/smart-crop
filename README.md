# Photo Processor

A Python script that processes photos to make them square (or custom ratio) and face-centered. It uses MTCNN for accurate face detection and handles EXIF orientation data.

## Features

- 📸 Process multiple images in parallel
- 👤 Accurate face detection using MTCNN
- 🔄 Automatic EXIF orientation correction
- 📐 Custom aspect ratio support (e.g., 16:9, 4:3)
- ⚡ Multi-processing for faster processing
- 💾 Preserves EXIF data in output images
- 🎯 Centers crop on detected faces
- 🔍 Visual face detection debugging

## Requirements

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - Pillow
  - mtcnn
  - piexif
  - tqdm
  - tensorflow
  - setuptools

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python crop.py
```

### Command Line Options

- `--input-dir`: Input directory containing images (default: "photos")
- `--output-dir`: Output directory for processed images (default: "output")
- `--ratio`: Crop ratio in format width:height (e.g., "16:9", "4:3"). Default is square
- `--show-faces`: Show face detection boxes and confidence scores on output images
- `--workers`: Number of worker processes (default: number of CPU cores)
- `--overwrite`: Overwrite existing files in output directory

### Examples

Process images with face detection visualization:
```bash
python crop.py --show-faces
```

Process images with custom aspect ratio:
```bash
python crop.py --ratio 16:9
```

Process images with all options:
```bash
python crop.py --input-dir my_photos --output-dir processed_photos --ratio 16:9 --show-faces --workers 4 --overwrite
```

## Directory Structure

```
.
├── photos/           # Input directory (default)
│   └── *.jpg        # Your photos to process
├── output/          # Output directory (default)
│   └── *.jpg        # Processed photos
├── crop.py
└── requirements.txt
```

## Notes

- The script processes JPG and PNG files
- Face detection confidence threshold is set to 95% to reduce false positives
- Images are automatically resized for faster face detection while maintaining accuracy
- EXIF orientation data is preserved in the output images
- Existing files are skipped by default unless `--overwrite` is used 