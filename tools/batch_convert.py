import os
import multiprocessing
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone

def convert_single_step_to_stl(args):
    """
    Converts a single STEP file to STL format.
    """
    step_filepath, stl_filepath = args
    
    
    if os.path.exists(stl_filepath):
        return f"Skipped (already exists): {os.path.basename(step_filepath)}"
    
    try:
       
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_filepath))
        
        if status != IFSelect_RetDone:
            return f"Failed to read: {os.path.basename(step_filepath)}"
            
        reader.TransferRoots()
        shape = reader.OneShape()

        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.SetASCIIMode(False) 
        writer.Write(shape, str(stl_filepath))
        
        return f"Success: {os.path.basename(step_filepath)}"
        
    except Exception as e:
        return f"Error on {os.path.basename(step_filepath)}: {str(e)}"

def main():

    INPUT_DIR = "./data/raw_step/fusion360" 
    OUTPUT_DIR = "./data/raw_step/fusion360_stl"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    step_files = list(Path(INPUT_DIR).glob("*.step")) + list(Path(INPUT_DIR).glob("*.stp"))
    print(f"Found {len(step_files)} STEP files to convert.")

    tasks = []
    for step_file in step_files:
        stl_filename = step_file.stem + ".stl"
        stl_filepath = os.path.join(OUTPUT_DIR, stl_filename)
        tasks.append((str(step_file), str(stl_filepath)))

    num_cores = multiprocessing.cpu_count()
    print(f"Starting conversion using {num_cores} cores...")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_single_step_to_stl, tasks), 1):
            if i % 100 == 0 or "Error" in result or "Failed" in result:
                print(f"[{i}/{len(step_files)}] {result}")

    print("Batch conversion completed!")

if __name__ == '__main__':
    main()