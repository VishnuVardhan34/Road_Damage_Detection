"""Create a small sample dataset to test the pipeline."""
import os
import random
from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET

def create_sample_dataset(num_samples=50):
    """Create synthetic road damage images with VOC XML annotations."""
    
    # Damage classes
    classes = ['D00', 'D10', 'D20', 'D40']  # Crack variations and Pothole
    class_names = {
        'D00': 'Longitudinal Crack',
        'D10': 'Transverse Crack',
        'D20': 'Alligator Crack',
        'D40': 'Pothole'
    }
    
    img_dir = 'data/raw/images'
    ann_dir = 'data/raw/annotations'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    print(f"Creating {num_samples} sample images with annotations...")
    
    for i in range(num_samples):
        # Create synthetic road image
        img = Image.new('RGB', (1280, 720), color=(80, 80, 90))  # Dark gray road
        draw = ImageDraw.Draw(img)
        
        # Add road texture
        for _ in range(500):
            x = random.randint(0, 1280)
            y = random.randint(0, 720)
            color = random.randint(70, 100)
            draw.point((x, y), fill=(color, color, color))
        
        # Number of damage instances
        num_damages = random.randint(1, 4)
        
        # Create XML annotation
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = 'images'
        ET.SubElement(annotation, 'filename').text = f'sample_{i:04d}.jpg'
        
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = '1280'
        ET.SubElement(size, 'height').text = '720'
        ET.SubElement(size, 'depth').text = '3'
        
        for j in range(num_damages):
            damage_class = random.choice(classes)
            
            # Random bbox
            xmin = random.randint(50, 1000)
            ymin = random.randint(50, 600)
            width = random.randint(50, 200)
            height = random.randint(30, 150)
            xmax = min(xmin + width, 1280)
            ymax = min(ymin + height, 720)
            
            # Draw damage on image
            if 'Crack' in class_names[damage_class]:
                # Draw crack
                for _ in range(10):
                    x1 = random.randint(xmin, xmax)
                    y1 = random.randint(ymin, ymax)
                    x2 = x1 + random.randint(-20, 20)
                    y2 = y1 + random.randint(-20, 20)
                    draw.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=2)
            else:
                # Draw pothole
                draw.ellipse([xmin, ymin, xmax, ymax], fill=(30, 30, 30), outline=(20, 20, 20))
            
            # Add to XML
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = damage_class
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        
        # Save image
        img.save(f'{img_dir}/sample_{i:04d}.jpg', quality=90)
        
        # Save XML
        tree = ET.ElementTree(annotation)
        tree.write(f'{ann_dir}/sample_{i:04d}.xml')
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{num_samples} samples...")
    
    print(f"\n✅ Created {num_samples} sample images!")
    print(f"Images: {img_dir}")
    print(f"Annotations: {ann_dir}")
    print(f"\nClass distribution:")
    for cls, name in class_names.items():
        print(f"  {cls}: {name}")

if __name__ == "__main__":
    create_sample_dataset(50)
