"""
Fusion 360 Assembly Dataset Extractor
=====================================
@Author: Di Liu
@Date: 2026-06-07
@Description: 
Extracts .step files and their corresponding global category labels
from the raw, unzipped Fusion 360 Assembly Dataset.

Usage: Configure the absolute paths below and run the script.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# Configuration (Use Absolute Paths)
# ==========================================
# Replace with the absolute path to your extracted Fusion 360 folder
# e.g., the folder containing '23495_6366269c', etc.
RAW_DATA_DIR = Path(r"D:\\Downloads\\fusion360") 

# Replace with the absolute path to your project's raw step folder
OUTPUT_STEP_DIR = Path(r"C:\\Users\\di_li\\Desktop\\SWL2026\\data\\raw_step\\fusion360")

# Replace with the absolute path where the label JSON should be saved
OUTPUT_LABEL_FILE = Path(r"C:\\Users\\di_li\\Desktop\\SWL2026\\data\\raw_step\\fusion360_labels.json")

def extract_dataset():
    print(f"Scanning raw directory: {RAW_DATA_DIR}")
    
    if not RAW_DATA_DIR.exists():
        print("Error: Raw data directory does not exist. Please check the path.")
        return

    # Ensure output directories exist
    OUTPUT_STEP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store filename -> category mapping
    global_labels = {}
    
    # Get all subdirectories (each represents an assembly)
    assembly_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(assembly_dirs)} assembly folders.")
    
    valid_parts_count = 0
    missing_json_count = 0

    # Process each assembly folder
    for asm_dir in tqdm(assembly_dirs, desc="Processing Assemblies"):
        json_path = asm_dir / "assembly.json"
        
        if not json_path.exists():
            missing_json_count += 1
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue # Skip corrupted JSONs
        
        # 1. Extract Global Category
        properties = data.get("properties", {})
        categories = properties.get("categories", [])
        
        # Skip if no category is assigned (we need ground truth for retrieval)
        if not categories:
            continue
            
        # Use the primary category as the label
        main_category = categories[0]
        
        # 2. Extract Bodies (.step files)
        bodies = data.get("bodies", {})
        for body_uuid, body_info in bodies.items():
            step_filename = body_info.get("step")
            
            if not step_filename:
                continue
                
            source_step_path = asm_dir / step_filename
            dest_step_path = OUTPUT_STEP_DIR / step_filename
            
            # Check if the .step file actually exists in the folder
            if source_step_path.exists():
                # Copy file to the consolidated folder
                shutil.copy2(source_step_path, dest_step_path)
                
                # Store label (using the filename without extension as the key)
                base_name = step_filename.replace('.step', '')
                global_labels[base_name] = main_category
                valid_parts_count += 1

    # 3. Save the Label Dictionary
    with open(OUTPUT_LABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(global_labels, f, indent=4)
        
    print("\n" + "="*50)
    print("Extraction Complete!")
    print(f"Successfully extracted {valid_parts_count} .step files with labels.")
    print(f"Labels saved to: {OUTPUT_LABEL_FILE}")
    if missing_json_count > 0:
        print(f"Skipped {missing_json_count} folders missing 'assembly.json'.")
    print("="*50)

if __name__ == "__main__":
    extract_dataset()