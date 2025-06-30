
import faiss   
import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from itertools import product
from collections import Counter

import ast
import gc
import hashlib
import subprocess

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps

import data_library as data_lib
import library as lib
from library import GlobalVars
from library import ExperimentConfig
   

def get_Faiss_metric_type(str_metric_type):
    metric_type = None
    if (isinstance(str_metric_type, str)):
        if (str_metric_type == 'L2' or 
            str_metric_type== 'l2' or
            str_metric_type== 'euclidian' or
            str_metric_type == 'manhattan'):
            metric_type = faiss.METRIC_L2
        elif (str_metric_type == 'cosine' or str_metric_type== 'correlation'):
            metric_type = faiss.METRIC_INNER_PRODUCT
    elif (isinstance(str_metric_type, int)):
        metric_type = str_metric_type
    return metric_type
        
def create_faiss_index(cfg):
    
    print("random seed was set to:", lib.RANDOM_SEED)
     
    """
    d = dimensiunea vectorilor (ex: 128)
    config = unul din dicționarul de mai sus
    """
    print("index params=", cfg._INDEX_PARAMS['params']['params'])
    
    
    
        
    metric_type = get_Faiss_metric_type(cfg._INDEX_PARAMS['metric_type'])
    print(f"Create FAISS index:{cfg._INDEX_PARAMS['params']['params']} with metric {metric_type}#{cfg._INDEX_PARAMS['metric_type']}")
    index_factory = cfg._INDEX_PARAMS['params']['params']
    if 'HNSW' in index_factory:
        print("set single thread")
        faiss.omp_set_num_threads(1)
    index = faiss.index_factory(cfg._FIELD_DIM, index_factory, metric_type)

        
    ef_construction = cfg._INDEX_PARAMS["params"].get("efConstruction", None)  
    if (not ef_construction is None):
        index.hnsw.efConstruction = ef_construction  
    
    return index
    
  

def predict_faiss_uniform(index, y_train, query_vectors, k=10):
    """
    Predict labels for query vectors using a FAISS index and majority vote from k nearest neighbors.

    Parameters:
        index (faiss.Index): FAISS index with all training vectors added.
        metadata_df (pd.DataFrame): DataFrame with metadata, must match the index order (row i = vector i).
        query_vectors (np.ndarray): Vectors to be predicted (shape: n_queries x dim).
        k (int): Number of nearest neighbors to consider.

    Returns:
        predictions (List): List of predicted labels (one per query).
        neighbors (List): Optional – List of k nearest labels for each query (for analysis).
    """
    metadata_df = pd.DataFrame({'label': y_train})
    label_array = metadata_df['label'].values
    
    distances, indices = index.search(query_vectors, k)
    neighbor_labels = label_array[indices]
    predictions = np.apply_along_axis(lib.majority_vote, axis=1, arr=neighbor_labels)

    return predictions.tolist(), indices.tolist()

def weighted_vote(labels, weights):
    unique_labels = np.unique(labels)
    vote_weights = np.zeros_like(unique_labels, dtype=float)

    for i, lbl in enumerate(unique_labels):
        vote_weights[i] = weights[labels == lbl].sum()

    return unique_labels[np.argmax(vote_weights)]

def predict_faiss_weighted(index, y_train, query_vectors, k=10, epsilon=1e-6):
    """
    Predict labels for query vectors using a FAISS index and weighted vote from k nearest neighbors.

    Parameters:
        index (faiss.Index): FAISS index with all training vectors added.
        metadata_df (pd.DataFrame): DataFrame with metadata, must match the index order (row i = vector i).
        query_vectors (np.ndarray): Vectors to be predicted (shape: n_queries x dim).
        k (int): Number of nearest neighbors to consider.

    Returns:
        predictions (List): List of predicted labels (one per query).
        neighbors (List): Optional – List of k nearest labels for each query (for analysis).
    """
    label_array = np.array(y_train)
    distances, indices = index.search(query_vectors, k)
    neighbor_labels = label_array[indices]

    weights = 1.0 / (distances + epsilon)

    predictions = []
    for lbls, wts in zip(neighbor_labels, weights):
        lbls = np.array(lbls)
        wts = np.array(wts)
        pred = weighted_vote(lbls, wts)
        predictions.append(pred)

    return predictions, indices.tolist()

def prepare_data_for_metric(data, metric_type):
    if (metric_type == 'cosine'):
        faiss.normalize_L2(data)
        return data
    if (metric_type == 'correlation'):
        data_centered = data - data.mean(axis=1, keepdims=True)
        faiss.normalize_L2(data_centered)
        return data_centered
    return data

def train_faiss_index(faiss_index, train_data, cfg):
    train_data = prepare_data_for_metric(train_data, cfg._INDEX_PARAMS['metric_type'])
    if not faiss_index.is_trained:
        faiss_index.train(train_data)
    faiss_index.add(train_data)    
    
def predict_with_faiss_index(faiss_index, test_data, y_train, vote_type, neighbors, cfg):
    test_data = prepare_data_for_metric(test_data, cfg._INDEX_PARAMS['metric_type'])
        
    predict_func = None
    if vote_type == "uniform":
        predict_func = predict_faiss_uniform
    elif vote_type == 'distance':
        predict_func = predict_faiss_weighted
    else:
        print("vote type not specified!")
    return predict_func(faiss_index, y_train=y_train, query_vectors=test_data, k=neighbors)

def get_faiss_index_file_name(cfg, suffix):
    ef_construction = cfg._INDEX_PARAMS["params"].get("efConstruction", None)  
    index_factory = cfg._INDEX_PARAMS['params']['params']
    index_file_name = cfg._COLLECTION_NAME + f"{index_factory}_{ef_construction}_{suffix}.index.faiss"
    return index_file_name
    
    
def get_faiss_index_file_path(cfg, suffix = ''):
    return GlobalVars.faiss_index_cache_path + get_faiss_index_file_name(cfg, suffix)
    
def cache_faiss_index(X_train, cfg):
    index =  create_faiss_index(cfg)
    
    if (cfg._NORMALIZE_VECTOR):
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        X_train_scaled = scaler.fit_transform(X_train)        
    else:
        X_train_scaled = X_train
    metrics = lib.measure_perf(train_faiss_index, index, X_train_scaled, cfg)
    
    index_file_path = get_faiss_index_file_path(cfg, cfg._INDEX_PARAMS['metric_type'])
    if (os.path.exists(index_file_path)):
        os.remove(index_file_path)
    
    faiss.write_index(index, index_file_path)
    metrics_file_path = f"{index_file_path}.metrics"
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f)
    
        
def execute_and_test_faiss(X_train, y_train, X_test, y_test, cfg):
    training_set_size = len(y_train)
    testing_set_size = len(y_test)
    
    
    
    if (cfg._NORMALIZE_VECTOR):
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        
    
    index_file_path = get_faiss_index_file_path(cfg, cfg._INDEX_PARAMS['metric_type'])
    print(f"use cache={cfg._USE_CACHE};file exists={os.path.exists(index_file_path)};path={index_file_path}")
    metrics_file_path = f"{index_file_path}.metrics"
    if (cfg._USE_CACHE and os.path.exists(index_file_path) and os.path.exists(metrics_file_path)):
        print("load index from cache:", index_file_path)
        index = faiss.read_index(index_file_path)
        with open(metrics_file_path, "r") as f:
            metrics = json.load(f)
        index_file_size = os.path.getsize(index_file_path) / 1024 / 1024
    else:
        index = create_faiss_index(cfg)    
        metrics = lib.measure_perf(train_faiss_index, index, X_train_scaled, cfg)
        index_file_path = get_faiss_index_file_path(cfg, cfg._ROW_KEY)
        faiss.write_index(index, index_file_path)
        index_file_size = os.path.getsize(index_file_path) / 1024 / 1024
        os.remove(index_file_path)
    
    gc.collect()    
    train_elapsed = metrics["elapsed_time"]
    train_memory = index_file_size
    
    
    nprobe = cfg._INDEX_PARAMS["params"].get("nprobe", None)        
    if (not nprobe is None):
        print(f"set nprobe={nprobe}", type(nprobe))
        index.nprobe = nprobe
        
    efSearch = cfg._INDEX_PARAMS["params"].get("efSearch", None)  
    if (not efSearch is None):
        print(f"set efSearch={efSearch}")
        index.hnsw.efSearch = efSearch  
    
    neighbors = cfg._INDEX_PARAMS["params"].get("neighbors", 10)        
    print("neighbors=", neighbors)
    vote_type = cfg._INDEX_PARAMS.get("index_type", "uniform")
    
        
    metrics = lib.measure_perf(
        predict_with_faiss_index,
        faiss_index = index,
        y_train=y_train,
        test_data=X_test_scaled,
        vote_type=vote_type,
        neighbors=neighbors,
        cfg=cfg        
    )
    predict_elapsed = metrics["elapsed_time"]
    predict_memory = metrics["peak_memory_MB"]
    y_pred, _ = metrics['result']
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"Acurracy FAISS: {accuracy * 100:.2f}%")
    metrics_json = {
        "neighbors": neighbors,
        "vote_type": vote_type,
        "accuracy_score": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "training_set_size":training_set_size,
        "testing_set_size": testing_set_size,
        "train_elapsed_time": train_elapsed,
        "predict_elapsed_time": predict_elapsed,
        "train_used_memory": train_memory,
        "predict_used_memory": predict_memory,
    }
    return metrics_json, y_pred


    
def execute_configuration(cfg):
    X_train, y_train, X_test, y_test = data_lib.load_training_and_testing_data_for_vectordb(cfg=cfg)
    
    metrics_json, y_pred = execute_and_test_faiss(X_train, y_train, X_test, y_test, cfg)
    return metrics_json


def execute_configuration_as_process(configuration, save_results = 0, row_index = None, display_std_out = True):
    if (type(configuration) == 'dict' or isinstance(configuration, dict)):
        params_str = json.dumps(configuration) 
    else:
        params_str = configuration
    
    #print("params as json=", params_str)
    
    md5_hash = hashlib.md5()
    md5_hash.update(params_str.encode('utf-8'))
    process_key = md5_hash.hexdigest()
    errors = None
    
    process_parameters = ["python", "faiss_experiment.py", process_key, str(save_results), params_str]
    #print("prpcess key=", process_key)
    process = subprocess.Popen(process_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()  # wait for finish 
    if (stdout != '' and display_std_out == True):
        print("stdout:", stdout)
        
    faiss_results = None
    error = None
    if (stderr == ''):        
        faiss_results_path = GlobalVars.faiss_results_path + f"{process_key}.faiss"
        with open(faiss_results_path, 'r') as json_file:
            faiss_results = json.load(json_file)
        #print("Result:", errors)
    else:
        print(f"Erori la configuratia {row_index}:", stderr)
        # logging.error(f"Process key: {process_key} with configs:\n{params_str}")
        # logging.error(stderr)
        error = stderr
    
    return process_key, faiss_results, error
 
 
def report_results(results_file_path, top_n):
    print(" ########## Reports ##############")
    df_sorted = lib.load_resuls_file(results_file_path)
    df_topn = df_sorted.head(top_n)

    cols_to_drop = ['skipped', 'training_set_size', 'testing_set_size', 'balanced_type']
    existing_cols = [col for col in cols_to_drop if col in df_topn.columns]
    df_topn = df_topn.drop(existing_cols, axis=1)
    df_topn_copy = df_topn.copy()
    df_topn_copy = df_topn_copy.drop(['row_key'], axis=1)    
    labels_desc = lib.get_queen_status_description_map()
    labels = [labels_desc[0], labels_desc[1], labels_desc[2], labels_desc[3]]

    vote_type_column = 'vote_type' if 'vote_type' in df_topn.columns else 'index_type'
    prev_acurracy_score = 0
    for index, row in df_topn.iterrows():
        process_key = row['row_key']    
        acurracy_score = row['accuracy']
        if (acurracy_score == prev_acurracy_score):
            continue
        prev_acurracy_score = acurracy_score
        faiss_results_path = GlobalVars.faiss_results_path + f"{process_key}.faiss"
        with open(faiss_results_path, 'r') as json_file:
            faiss_results = json.load(json_file)
            
            
        title = f"Config: segment({row['segment_lenght']}, {row['segment_overlap']}), feat({row['feature']}, {row['vector_operation']}), metric({row['metric_type']},{row['index_params']}, {row[vote_type_column]}) \nAccuracy {faiss_results['accuracy_score']*100:.2f}\n"
        print(title)
        cm = np.array(faiss_results['confusion_matrix'])
        classification_report = faiss_results['classification_report']
        lib.display_confusion_matrix(cm)
        lib.display_classification_report(classification_report)
     
     
     
def excute_esemble_config(cfg):
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)

    test_records = data_lib.load_and_prepare_data_for_vector_db(cfg, "test")    
    X_test = test_records['vector_data']
    y_test = test_records['queen_status']
    X_test_ok = np.array(X_test.tolist())

    train_records = data_lib.load_and_prepare_data_for_vector_db(cfg)
    X_train1 = train_records['vector_data']
    y_train1 = train_records['queen_status']
    X_train1_ok = np.array(X_train1.tolist())
    
    # scaler = MinMaxScaler(feature_range=(0, 1)) 
    # X_train_scaled = scaler.fit_transform(X_train1_ok)
    # X_test_scaled = scaler.transform(X_test_ok) 

    #print("train scalled shape:", X_train_scaled.shape)
    # enn = EditedNearestNeighbours(n_neighbors=3)
    # X_train_ok, y_train = enn.fit_resample(X_train_scaled, y_train1)

    # ncl = NeighbourhoodCleaningRule(n_neighbors=3, n_jobs=-1)
    # X_train_ok, y_train = ncl.fit_resample(X_train_scaled, y_train1)
    # tl = TomekLinks(sampling_strategy='auto')  # auto = toate clasele
    # X_train_ok, y_train = tl.fit_resample(X_train_scaled, y_train1)

    # print("after resample shape:", X_train_ok.shape)
    # cfg._SEGMENT_OVERLAP=2
    # processed_records = data_lib.load_and_prepare_data_for_vector_db(cfg)
    # X_train2 = processed_records['vector_data']
    # y_train2 = processed_records['queen_status']
    # X_train2_ok = np.array(X_train2.tolist())
    
    # cfg._SEGMENT_OVERLAP=0
    # processed_records = data_lib.load_and_prepare_data_for_vector_db(cfg)
    # X_train3 = processed_records['vector_data']
    # y_train3 = processed_records['queen_status']
    # X_train3_ok = np.array(X_train3.tolist())
    
    X_train_ok = X_train1_ok
    y_train = y_train1

    # X_train_ok = np.concatenate((X_train1_ok, X_train2_ok, X_train3_ok))
    # y_train = np.concatenate((y_train1, y_train2, y_train3))

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    X_train_scaled = scaler.fit_transform(X_train_ok)
    X_test_scaled = scaler.transform(X_test_ok) 

    faiss_indexes = []
    train_class_labels = []

    index =  faiss_tool.create_faiss_index(cfg)
    index.add(X_train_scaled)
    binary_predictions = []
    distances = []
    neighbors_no = []
    accuracies_per_class = []
    n_classes = 4
    k=15
    for cls in range(n_classes):
        print(f"process for class: {cls}")
        y_train_binary = (y_train == cls).astype(int)
        y_test_binary = (y_test == cls).astype(int)
        # y_test_binary = (y_test == i).astype(int)

        cls_mask = (y_train == cls)
        local_pred, local_neighbors_count, local_distances = faiss_tool.predict_faiss(index, y_train=y_train_binary, query_vectors=X_test_scaled)
        binary_predictions.append(local_pred)
        distances.append(local_distances)
        neighbors_no.append(local_neighbors_count)
        accuracies_per_class.append(accuracy_score(y_test_binary, local_pred))

    binary_predictions = np.stack(binary_predictions,  axis=1)
    distances = np.stack(distances,  axis=1)
    neighbors_no = np.stack(neighbors_no,  axis=1)
    final_preds = []
    print("bin preds=", binary_predictions)
    print("shape bine:", binary_predictions.shape[0])
    print("shape distances:", len(distances))
    print(accuracies_per_class)
    count_more = 0
    count_zero = 0
    for i in range(binary_predictions.shape[0]):
        preds_1 = binary_predictions[i] == 1
        if np.sum(preds_1) == 1:
            # Doar o clasă a spus "da"
            final_preds.append(np.argmax(preds_1))
        elif np.sum(preds_1) > 1:
            # Mai multe clase au spus "da" → alegem după distanță
            competing_dists = distances[i][preds_1]
            competing_classes = np.where(preds_1)[0]
            best_class = competing_classes[np.argmin(competing_dists)]
            final_preds.append(best_class)
            count_more += 1
        else:
            # Nicio clasă nu a spus "da" → fallback (opțional)
            # Alegi clasa cu cea mai mică distanță
            final_preds.append(np.argmax(neighbors_no[i]))
            count_zero += 1

    print(f"we have {count_more} predictions with more classes and {count_zero} with zero classes")
    #y_pred - final
    #y_pred = np.argmax(np.array(predictions), axis=0)  # Alegem clasa cu cele mai multe voturi pentru fiecare exemplu
    #y_pred = np.argmin(np.array(distances), axis=0)
    y_pred = np.array(final_preds)

    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    training_set_size = len(y_train)
    testing_set_size = len(y_test)
    
    metrics_json = {
        "accuracy_score": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "training_set_size":training_set_size,
        "testing_set_size": testing_set_size
    }

    print(f"Acurracy Ensemble : {accuracy * 100:.2f}%")

        # cfg._NORMALIZE_VECTOR = False
        # metrics_json, _ = faiss_tool.execute_and_test_faiss(X_train_scaled, y_train_binary, X_test_scaled, y_test_binary, cfg)
    return metrics_json


def predict_faiss_with_validity(index, y_train, query_vectors, validity_scores, k=15, threshold=0.6):
    """
    Predict labels for query vectors using a FAISS index and majority vote from k nearest neighbors.

    Parameters:
        index (faiss.Index): FAISS index with all training vectors added.
        metadata_df (pd.DataFrame): DataFrame with metadata, must match the index order (row i = vector i).
        query_vectors (np.ndarray): Vectors to be predicted (shape: n_queries x dim).
        k (int): Number of nearest neighbors to consider.

    Returns:
        predictions (List): List of predicted labels (one per query).
        neighbors (List): Optional – List of k nearest labels for each query (for analysis).
    """
    metadata_df = pd.DataFrame({'label': y_train})
    
    distances, indices = index.search(query_vectors, k)
    predictions = []
    neighbors_no = []
    neighbors_distances = [] 
    epsilon = 1e-6


    for i, (neighbor_indices, dists) in enumerate(zip(indices, distances)):
        # Obține etichetele vecinilor
        neighbor_labels = metadata_df.iloc[neighbor_indices]['label'].values
        neighbor_validity = validity_scores[neighbor_indices]

        mask = neighbor_validity >= threshold
        filtered_labels = neighbor_labels[mask]
        filtered_dists = dists[mask]

        if len(filtered_labels) == 0:
            # fallback: vot majoritar pe toți vecinii (sau distanță minimă)
            vote = Counter(neighbor_labels).most_common(1)[0][0]
        else:
            vote = Counter(filtered_labels).most_common(1)[0][0]
        predictions.append(vote)

    #     count_same_class = np.sum(np.array(neighbor_labels) == vote)
    #     neighbors_no.append(count_same_class)

    #     dists = distances[i]
    #     vote_dists = dists[neighbor_labels == vote]
    #     vote_dist_mean = vote_dists.mean() if len(vote_dists) > 0 else float('inf')
    #     neighbors_distances.append(vote_dist_mean)
    # return predictions, neighbors_no, neighbors_distances
    return predictions

def compute_validity_scores(index, X_train, y_train, k):
    distances, indices = index.search(X_train, k + 1)
    indices = indices[:, 1:]  # shape: (n_samples, k)
    validity_scores = []
    #metadata_df = pd.DataFrame({'label': y_train})
    for i in range(X_train.shape[0]):

        neighbors = indices[i]
        own_label = y_train.iloc[i]
        neighbor_labels = y_train.iloc[neighbors].values
        same_class = np.sum(neighbor_labels == own_label)
        validity = same_class / k
        validity_scores.append(validity)

    validity_scores = np.array(validity_scores)
    return validity_scores

def cache_all_faiss_indexes(experiments_file_path):
    df = pd.read_csv(experiments_file_path)
    fields_names = ["segment_lenght","segment_overlap","feature","vector_operation","metric_type", "indexFactory_full2"]
    
    #df['index_unique'] = df[df["segment_lenght"] + "_" + df["segment_overlap"] + "_" + df["feature"] + "_" + df["vector_operation"] + "_" + df["metric_type"]]
    #df['indexFactory'] = df['index_params'].apply(lambda item: ast.literal_eval(item).get("params"))
    df['indexFactory'] = df['index_params'].apply(lambda item: ast.literal_eval(item).get("params"))
    df['efConstruction'] = df['index_params'].apply(lambda item: str(ast.literal_eval(item).get("efConstruction")))
    df['indexFactory_full2'] = df['indexFactory'] + "-" + df['efConstruction']
    
    
    df_unique = df_unique = df.groupby(fields_names, as_index=False).first()
    df_unique = df_unique.reset_index(drop=True)
    
    top_n = None
    if (not top_n is None):
        df_unique = df_unique.head(top_n)
    total_steps = len(df_unique)
    with tqdm(total=total_steps, desc="Evaluating configs") as pbar:
        for index, row in df_unique.iterrows():
            cfg = ExperimentConfig(row)
            GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    
            X_train, y_train, X_test, y_test = data_lib.load_training_and_testing_data_for_vectordb(cfg=cfg)
            cache_faiss_index(X_train=X_train, cfg=cfg)            
            pbar.update(1) 
            
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        os.makedirs(GlobalVars.faiss_index_cache_path, exist_ok=True)
        os.makedirs(GlobalVars.faiss_results_path, exist_ok=True)
        process_key = sys.argv[1]
        save_results = sys.argv[2]        
        configs_json = json.loads(sys.argv[3])  #json decode
        print("configs json=", configs_json)
        random_seed = configs_json.get("random_seed", 0)
        print(f"Param random seed: {random_seed}")
        
        lib.init_random_seed(random_seed)
        
        cfg = ExperimentConfig(configs_json)
        cfg._ROW_KEY = process_key
        GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    
        faiss_results = execute_configuration(cfg)
        print(faiss_results)
        #logging.info(f"Process key: {process_key} with configs:\n{sys.argv[3]}")
        
        #errors save always
        faiss_results_path = GlobalVars.faiss_results_path + f"{process_key}.faiss"
        with open(faiss_results_path, 'w') as json_file:
            json.dump(faiss_results, json_file, indent=4)
        exit(0)
    else:
        print("No params!")
        exit(1)