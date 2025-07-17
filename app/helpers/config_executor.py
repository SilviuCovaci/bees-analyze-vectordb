import pandas as pd
import numpy as np

import threading
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock
import random
from tqdm import tqdm

from library import GlobalVars
from library import ExperimentConfig
import library as lib

progress_bar: tqdm = None
write_lock = None
progress_lock = None

execute_probability = 1  
debug_mode = False

def execute_one_row_from_configurations_from_lib(row, exec_tool):
    lib.init_ramdom_seed(lib.RANDOM_SEED)
    cfg_dict = row.to_dict()
   
    cfg = ExperimentConfig(cfg_dict)
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    return  exec_tool.execute_configuration(cfg)    
    
def execute_one_row_from_configurations_as_process(row, exec_tool):
    global execute_probability, write_header, network_type
    
    new_row = row.copy()
    if random.random() > execute_probability:        
        new_row["skipped"] = True
        return new_row
            
    cfg_dict = new_row.to_dict()      
    process_key, faiss_results, error =  exec_tool.execute_configuration_as_process(cfg_dict, row_index=str(row.name), display_std_out=debug_mode)
    

    if (faiss_results is not None):
        new_row["skipped"] = False
        new_row['accuracy'] = faiss_results["accuracy_score"]
        new_row['training_set_size'] = faiss_results["training_set_size"]
        new_row['testing_set_size'] = faiss_results["testing_set_size"]
        new_row['row_key'] = process_key
    else:
        new_row["skipped"] = True
        print("Error on Faiss", error)
    return new_row

def init_locks(total_rows):
    global write_lock, progress_lock, progress_bar
    write_lock = threading.Lock()
    progress_lock = threading.Lock()

    if (progress_bar is not None):
        progress_bar.close()        
    progress_bar = tqdm(total=total_rows, desc="Process", unit="rows")
    
def process_chunk(chunk, output_file_path, exec_tool):    
    global write_lock, progress_lock, progress_bar
    
    processed_rows = chunk.apply(lambda row: launch_execute_configurations_as_process(exec_tool=exec_tool, row=row, debug_mode=debug_mode), axis=1) 
    processed_rows = processed_rows[processed_rows["skipped"] == False]
    try:
        with write_lock:
            processed_rows[lib.result_fields()].to_csv(output_file_path, mode='a', header=False, index=False)
                
    except Exception as ex:
        print("ex:", ex)
    with progress_lock:  
        progress_bar.update(len(chunk))    
        
def process_all_configs_by_threads(input_file_path, output_file_path, exec_tool, num_threads = 10, chunk_size = 200):
    global write_lock, progress_lock, progress_bar

    with open(input_file_path, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # Scade 1 pentru header
        
    init_locks(total_rows)

    #df_sample = pd.read_csv(input_file_path, nrows=0)
    df_sample = pd.DataFrame(columns=lib.result_fields())
    df_sample.to_csv(output_file_path, mode="w", index=False)  # Scriem doar header-ul 
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunkId = 0
        for chunk in pd.read_csv(input_file_path, chunksize=chunk_size):
            chunkId += 1
            #print(f"Started chunk: {chunkId}")
            executor.submit(lambda chunk=chunk: process_chunk(chunk, output_file_path=output_file_path, exec_tool=exec_tool))
    
    if (progress_bar):
        progress_bar.close()
        progress_bar = None
    print("Done")
    
def launch_execute_configurations_as_process(exec_tool, row, execute_probability = 1, do_normalize = 1, debug_mode=True, is_knn = False):    
    
    new_row = row.copy()
    if random.random() > execute_probability:        
        new_row["skipped"] = True
        return new_row
            
    cfg_dict = new_row.to_dict()      
    cfg_dict["normalize"] = do_normalize
    cfg_dict['balanced_type'] = 1

    process_key, tool_results, error =  exec_tool.execute_configuration_as_process(cfg_dict, save_results=0, row_index=str(row.name), display_std_out=debug_mode)

    new_row = lib.extract_experiment_results(new_row=new_row, tool_results=tool_results, process_key=process_key, error=error)
    return new_row