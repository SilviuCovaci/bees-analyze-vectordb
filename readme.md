# 🐝 BeeHive Acoustic Analysis with VectorDB

This repository contains the source code for the dissertation project titled  
**"Utilizarea bazelor de date vectoriale pentru monitorizarea acustică a coloniilor de albine"**.

---

## 📦 Clone this repository

```bash
git clone https://github.com/SilviuCovaci/bees-analyze-vectordb.git
cd bees-analyze-vectordb
```

## 📄 Project Overview

This project explores the use of acoustic analysis and vector similarity search to detect the presence of the queen bee in a beehive.

The workflow includes:
- audio preprocessing and segmentation,
- feature extraction using Mel-Frequency Cepstral Coefficients (MFCC),
- vector aggregation (mean, IQR-based, etc.),
- and classification using distance-based methods such as K-Nearest Neighbors (KNN), FAISS vector search and Milvus vector search.

All implementation details follow the methodology described in the dissertation.

## 🔧 Requirements

The project requires Python 3.11 and the following key libraries:

- `numpy` – numerical operations
- `librosa` – audio signal processing
- `pandas` – data manipulation
- `scikit-learn` – machine learning models and evaluation
- `faiss-cpu` – vector indexing and similarity search
- `matplotlib` or `seaborn` – (optional) for visualization
- `jupyter` – for running notebooks

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Follow the steps below to reproduce the workflow:

## 📥 Download Dataset

The dataset is not included in this repository but it can be downloaded from following location:
https://www.kaggle.com/datasets/annajyang/beehive-sounds


You should manually download the audio dataset and place it in the `dataset/` directory.

> 📁 Recommended structure:
> ```
> dataset/
> └── soundfiles/
>     ├── 2022-06-05--17-41-01_2__segment0.wav
>     ├── 2022-06-05--17-41-01_2__segment1.wav
>     └── ...
> all_data_updated.csv
> ```

## ⚙️ Data Preprocessing and Feature Extraction

The preprocessing pipeline involves a few key steps  
*(see Jupyter notebook: `preprocess_data.ipynb`)*:


### ✅ Step 1 – Synchronize metadata with available audio files

This step prepares the initial metadata table (`all_data_updated.csv`) by:
- cleaning unused columns,
- identifying available audio files,
- assigning descriptive labels,
- creating train/test split,
- and saving a new file: `all_data_sync.csv`  with 7,100 records,

```python
synch_df = lib.sync_metadata_with_audio(save_processed=True, show_df_stats=False)
```

### ✅ Step 2 – Generate segmentation metadata

This step:
- expands the synchronized metadata to reflect subsegment information based on desired segmentation parameters (e.g., 10s segments with 5s overlap).
- calculates start and end times per segment,
- generates a segmentation folder like: `extended_segments_10_sec_overlap_5`,
- in segmentation folder it creates an extended CSV for all subsegments, called `dataset_subsegments.csv`,


You can Generate segmentation metadata using:

```python
GlobalVars.set_segment_lenght_and_overlap(10,5)
df=pd.read_csv(GlobalVars.csv_data_sync_path )
lib.generate_segmentation_metadata(df, segment_length=GlobalVars.segment_length, overlap=GlobalVars.overlap, recreate_if_exists=False)
```
### ✅ Step 3 -  Feature Extraction

Feature extraction is performed on the previously segmented metadata. You can define which features to extract by editing the `features` list.

In this implementation, audio features are extracted and stored in SQLite files, one per partition. Processing is parallelized using Dask, and each partition is written independently before being merged into a single dataset, as final step.

Example:

```python
features = ['pe-mfcc_40']

manager = multiprocessing.Manager()
global_config = manager.dict({
    "segment_lenght": GlobalVars.segment_length,
    "overlap": GlobalVars.overlap,
    "features": features
})

# Define input CSV path (generated during segmentation step)
datasetpath = GlobalVars.extended_dataset_file_path()

# Run feature extraction with parallelized Dask processing
dask_helper.process_csv(
    datasetpath,
    one_row_processor=None,
    partition_processor=lambda batch, partition_info:
        sql_lib.extract_features_from_batch(
            df_partition=batch,
            partition_info=partition_info,
            global_config=global_config
        ),
    blocksize='500KB'
)

# Merge all per-partition SQLite files into a single database
data_lib.merge_all_partitions(features)
```

After processing, the output will be saved in the segmentation folder as a single .db file per feature. For example:

> ```
> extended_segments_10_sec_overlap_5/
> ├── dataset_subsegments.csv
> ├── pe-mfcc_40_features.db
> └── ...
> ```

### ✅ Step 4 - Vector Aggregation & Caching

To speed up vector-based experiments, the extracted features are aggregated and saved in precomputed **cache files**.

Each cache file is defined by:
- a segmentation configuration (e.g., 10s with 5s overlap),
- one or more features (e.g., `pe-mfcc_40`),
- and an aggregation method (e.g., `mean_iqr25`, `mean`, `mean_iqr15`).

**Example: Caching a single configuration**

```python
"""
Test caching for one specified configuration
"""

cache_config_dict = {
    "segment_lenght": 10,
    "segment_overlap": 5,
    "feature": "pe-mfcc_40",
    "vector_operation": "mean_iqr25"
}

cache_cfg = ExperimentConfig(cache_config_dict)
data_lib.cache_one_configuration(cache_cfg)
```
The resulting cache file will be saved in the segmentation folder, for example:
> ```
> extended_segments_10_sec_overlap_5/
> ├── dataset_subsegments.csv
> ├── pe-mfcc_40_features.db
> ├──__vectors_pe-mfcc_40_mean_iqr25.cache
> └── ...
> ```
You can automatically generate and cache multiple combinations of segmentations, features, and aggregation strategies:

**Example: Caching multiple configurations**

⚠️ Important: Before calling cache_all_collections(), make sure that all required features
(e.g. pe-mfcc_40) have already been extracted and saved as .db files in the right segmentation folder.
Otherwise, caching will fail due to missing input data.

```python
"""
Test caching for more configurations once
"""
data_lib.combine_all_data_parameters()
data_lib.cache_all_collections()
```


Each generated .cache file contains the vector representation (after aggregation) for all subsegments and is used in later classification steps.


## 🧪 Experiments and Evaluation

All experiments are grouped by method: **KNN**, **FAISS**, and **Milvus**.  
Each method has its own experiment logic and result analysis tools.

### 🔹 Structure

For each method, the following components are available:

- `*_experiment.py`: script to run the experiment on a given configuration.
- `*_playground.ipynb`: Jupyter notebook for interactive testing and parameter tuning.
- `*_results_analyze.ipynb`: notebook to load results and visualize/analyze performance metrics.

### 🔄 Experiment Execution Modes

Each experiment can be executed in two ways:

- **As a standalone process**, by running the corresponding Python script with a set of parameters (e.g., via CLI).
- **As a library call**, by importing and invoking the experiment function directly from another script or notebook.

When running multiple experiments in parallel (e.g., different configurations or datasets), each experiment is executed as an **independent process**, allowing parallel execution across multiple CPU cores.

This hybrid design offers both flexibility (for interactive testing) and scalability (for batch execution).


### 🔧 KNN - Experiment Preparation

Before running KNN experiments, a set of configuration combinations is generated.  
These include variations of parameters such as distance metric, number of neighbors, feature aggregation method, and segmentation settings.

The list of experiments is saved to a CSV file (e.g., `all_knn_experiments_v3.csv`), and the results of executed experiments will be appended to another file.

If needed, you can customize which combinations are generated by modifying the logic inside the `combine_knn_multiple_v3()` function from `knn_experiment.py`.

Example:

```python
all_knn_experiments_file_path = GlobalVars.experiments_path + "all_knn_experiments_v3.csv"
knn_experiments_output_file = GlobalVars.experiments_path + "executed_knn_configs_results_v3.csv"

# Generate all experiment combinations
knn_tool.combine_knn_multiple_v3(all_knn_experiments_file_path)
```

### ▶️ KNN – Running a Single Experiment

You can run a single experiment configuration in two ways: as an individual **process** or as a direct **library function call**.

---

#### 🔹 Option 1 – Run as a separate process

```python
# Execute ONE configuration by launching it as an independent process
cfg_records = pd.read_csv(all_knn_experiments_file_path)
row = cfg_records.iloc[0].copy()

new_row = executor.launch_execute_configurations_as_process(
    knn_tool,
    row,
)
```
The result represents a row from the results CSV file and includes evaluation metrics such as accuracy and per-class precision, as well as performance measurements for both the training and prediction steps.

#### 🔹 Option 2 – Run directly via library call

```python
# Execute ONE configuration by calling the experiment function directly
cfg_records = pd.read_csv(all_knn_experiments_file_path)
row = cfg_records.iloc[0].copy()
cfg = ExperimentConfig(row)
GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
knn_results = knn_tool.execute_configuration(cfg)
```
The knn_results is a dictionary that includes detailed evaluation metrics, such as the confusion matrix and a full classification report.

### 🔁 Running All KNN Experiments (with Repeats)

You can run all experiment configurations from the CSV file in multiple rounds, e.g. for statistical analysis or robustness testing.

```python
# Execute all configurations from the generated experiment file
# Repeating the full experiment N times

repeat_no = 5

for i in range(repeat_no):
    print(f"Execute round {i}")

    knn_experiments_output_file = GlobalVars.experiments_path + \
        f"executed_knn_configs_results_v3_round{i}.csv"

    executor.process_all_configs_by_threads(
        exec_tool=knn_tool,
        input_file_path=all_knn_experiments_file_path,
        output_file_path=knn_experiments_output_file,
        num_threads=4,
        chunk_size=20
    )
```
Each round will process all configurations in parallel (using threads) and save the results to a separate output file (one per round).

> 🔄 **Note:** To extend the dataset across all segmentations of the same duration, you can use the special segmentation format `(10, all)`.  
> This will combine all available segmentations with a length of 10 seconds, regardless of overlap, into a single dataset. This is useful for increasing sample diversity and improving generalization.


### ⚡ FAISS Experiments

FAISS is used for efficient vector similarity search using index structures such as `Flat`, `IVF`, `PQ`, and `HNSW`.

Compared to KNN, FAISS experiments are more complex, as they involve additional indexing-specific parameters.  
Each experiment combines dataset-related parameters (features, segmentations, aggregations) with index-specific configurations (e.g., `nlist`, `nprobe`, `M`, `nbits`).

---

### 🔧 Preparing FAISS experiment configurations

Experiments are generated **per index type**. For each type (e.g., `ivf`, `flat`, `hnsw`), a predefined list of index parameters is available. These can be extended or customized in the code.

Example: generating all experiments for IVF indexes.

```python
index_type = "ivf_all"

faiss_experiments_output_file = GlobalVars.experiments_path + \
    f"executed_faiss_experiments_index_{index_type}.csv"

all_faiss_experiments_file_path = GlobalVars.experiments_path + \
    f"all_faiss_experiments_index_{index_type}.csv"

lib.combine_multiple_parameters_v3(
    all_faiss_experiments_file_path,
    ex_cfg.ivf_all  # predefined list of IVF-specific index configurations
)
```
🛠️ You can customize or extend the list of index parameters in the ex_cfg.<index_type> dictionary.

### ▶️ FAISS – Running Experiments

FAISS experiments can be executed in a similar way to KNN, using three approaches:

1. **Single configuration as a separate process**  
2. **Single configuration as a direct function call**  
3. **Multiple configurations in parallel**, by reading from a CSV file

---

#### 🔹 1. Run one configuration as a separate process

```python
# Execute ONE configuration by launching it as an independent process
cfg_records=pd.read_csv(all_faiss_experiments_file_path)
row = cfg_records.iloc[0].copy()
new_row = executor.launch_execute_configurations_as_process(
    faiss_tool,
    row,
)
```
The result represents a row from the results CSV file and includes evaluation metrics such as accuracy and per-class precision, as well as performance measurements for both the training and prediction steps.

#### 🔹 2. Run one configuration via function call

```python
# Execute ONE configuration by calling the experiment function directly
cfg_records=pd.read_csv(all_faiss_experiments_file_path)
row = cfg_records.iloc[0].copy()
cfg = ExperimentConfig(row)
GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
faiss_results = faiss_tool.execute_configuration(cfg)
```

The `faiss_results` is a dictionary that includes detailed evaluation metrics, such as the confusion matrix and a full classification report.

### 🔁 Running All FAISS Experiments (with Repeats)

You can run all experiment configurations from the CSV file in multiple rounds, e.g. for statistical analysis or robustness testing.

```python
# Execute all configurations from the generated experiment file
# Repeating the full experiment N times

repeat_no = 5
index_type = 'ivf_all'

all_faiss_experiments_file_path = GlobalVars.experiments_path + f"all_faiss_experiments_index_{index_type}.csv"
lib.combine_multiple_parameters_v3(all_faiss_experiments_file_path, ex_cfg.ivf_all)

for i in range(repeat_no):
    print(f"Execute round {i}")

    #you can cache the indexes before prediction
    #cache_all_faiss_indexes(all_faiss_experiments_file_path)
    
    faiss_experiments_output_file = GlobalVars.experiments_path + f"executed_faiss_experiments_index_{index_type}_round{i}.csv"
    executor.process_all_configs_by_threads(exec_tool=faiss_tool, input_file_path=all_faiss_experiments_file_path, output_file_path=faiss_experiments_output_file, num_threads = 4, chunk_size = 20)
```
Each round will process all configurations in parallel (using threads) and save the results to a separate output file (one per round).

> 🔄 **Note:** To extend the dataset across all segmentations of the same duration, you can use the special segmentation format `(10, all)`.  
> This will combine all available segmentations with a length of 10 seconds, regardless of overlap, into a single dataset. This is useful for increasing sample diversity and improving generalization.