import dask
import dask.dataframe as dd
import dask.bag as db
from dask.diagnostics import ProgressBar
import time

dask_pbar = None

dask.config.set({'dataframe.query-planning': False})

def process_csv(file_path,  one_row_processor, partition_processor = None, blocksize = 10000, dask_scheduler='processes', meta = None, dtype=None):
    #read csv file using dask
    ddf = dd.read_csv(file_path,  blocksize = blocksize, dtype = dtype)

    if (meta is None):
        print("Set meta from csv")
        meta = ddf
        
    df_processed = ddf.map_partitions(lambda partition: partition.apply(one_row_processor, axis=1), meta=meta)    
    if (partition_processor is None):
        df_processed = ddf.map_partitions(lambda partition: partition.apply(one_row_processor, axis=1), meta=meta)
    else:         
        df_processed = ddf.map_partitions(partition_processor, meta=meta)
            
    with ProgressBar():
        processed_records = df_processed.compute(scheduler=dask_scheduler)    
    print(f"Process finalizat")
    return processed_records

def process_dataframe(df, one_row_processor, partition_processor = None, npartitions = 10, dask_scheduler='processes', meta=None):
    # Convertim pandas.DataFrame în dask.DataFrame
    ddf = dd.from_pandas(df, npartitions=npartitions)

    if meta is None:
        meta = ddf.head(1)

    if (partition_processor is None):
        df_processed = ddf.map_partitions(lambda partition: partition.apply(one_row_processor, axis=1), meta=meta)
    else: 
        df_processed = ddf.map_partitions(partition_processor, meta=meta)
        
    with ProgressBar():
        processed_records = df_processed.compute(scheduler=dask_scheduler)

    return processed_records


def process_records_in_paralel(records,  partition_processor, npartitions = 10, dask_scheduler='processes'):
    dask_bag = db.from_sequence(records.to_dict(orient="records"), npartitions=npartitions)
    with ProgressBar():
        results = dask_bag.map_partitions(
            lambda batch: partition_processor(batch=batch)
        ).compute(scheduler=dask_scheduler)
    return results
    


def fake_process_one_row(row):
    time.sleep(0.001)
    return row

def benchmark_blocksize(file_path, blocksize, dask_scheduler):
    start_time = time.time()
    process_csv(file_path, fake_process_one_row, blocksize=blocksize, dask_scheduler=dask_scheduler) 
    elapsed_time = time.time() - start_time
    return elapsed_time

def benchmark_blocksizes(file_path, dask_scheduler="processes"):
    blocksizes = ["100KB", "500KB", "1MB", "2MB", "5MB", None]  # Testăm diferite valori
    results = {bs: benchmark_blocksize(file_path, bs, dask_scheduler=dask_scheduler) for bs in blocksizes}

    for bs, t in results.items():
        print(f"Blocksize {bs}: {t:.2f} sec")
        
    return results