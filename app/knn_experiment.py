import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import json
import os
from itertools import product
import random
import hashlib
import subprocess

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import time

import data_library as data_lib
import library as lib
from library import GlobalVars
from library import ExperimentConfig
   
def combine_knn_multiple_parameters():
    # segment_prop = [(5,0), (5,2), (7, 5), (10, 0), (10,5)]
    # features = ['mfcc_20', 'mfcc_30', 'mfcc_40', 'mfcc_50']
    # operations = ['mean', 'mean_iqr15', 'mean_iqr25']
    # metric_type = ['euclidean', 'manhattan', 'cosine', 'correlation']
    # weights = ['uniform', 'distance']
    # neighbors = [7, 10, 15, 20]
    
    
    segment_prop = [(7, 5), (10, 0), (10,5)]
    features = ['pe-mfcc_40', 'pe-mfcc_50']
    operations = ['mean', 'mean_iqr15', 'mean_iqr25']
    metric_type = ['cosine', 'correlation']
    weights = ['uniform', 'distance']
    neighbors = [10, 15, 20]
    
    
    #all posible combinations
    param_combinations = list(product(
        segment_prop, features, operations, metric_type, weights, neighbors
    ))

    all_configurations = [
        {
            "segment_lenght": segp[0],            
            "segment_overlap": segp[1],        
            "feature": feat,
            "vector_operation": operation,     
            "metric_type": m_type,
            "index_type": i_type, #weights
            "index_params": i_param, #neighbors
            "skipped": False,
            "accuracy": None,
            "training_set_size": None,
            "testing_set_size": None,
            "row_key": None,   
        }
        for segp, feat, operation, m_type, i_type, i_param in param_combinations
    ]
    df_configuration = pd.DataFrame(all_configurations)
    df_configuration.to_csv('experiments' + os.sep + 'all_knn_experiments.csv', index=False) 

def combine_knn_multiple_v3(combinations_file_path):
    segment_prop = [(10,'all')]
    features = ['pe-mfcc_40']
    operations = ['mean', 'mean_iqr25']
    metric_types = ['cosine', 'correlation']
    vote_types = ['uniform', 'distance']
    neighbors_list = [15, 20]
    
    
    #all posible combinations
    param_combinations = list(product(
        segment_prop, features, operations, metric_types, vote_types, neighbors_list
    ))    

    #metric_type,vote_type,neighbors,index_params,

    all_configurations = [
        {
            "segment_lenght": segp[0],            
            "segment_overlap": segp[1],        
            "feature": feat,
            "vector_operation": operation,     
            "metric_type": m_type,
            "vote_type": vote_type, #weights
            "neighbors": neighbors,
            "index_params": '{"neighbors":' + str(neighbors) + '}',
            "skipped": False,
            "accuracy": None,
            "precision_0": None,
            "precision_1":None,
            "precision_2":None,
            "precision_3":None,
            "training_set_size": None,
            "testing_set_size": None,
            "train_time": None,
            "predict_time": None,
            "train_used_memory": None,
            "predict_used_memory": None,            
            "row_key": None,   
        }
        for segp, feat, operation, m_type, vote_type, neighbors in param_combinations
    ]
    df_configuration = pd.DataFrame(all_configurations)
    df_configuration.to_csv(combinations_file_path, index=False) 
    
def load_training_data_for_knn(cfg):
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    processed_records = data_lib.load_and_prepare_data_for_vector_db(cfg)
    # print(processed_records.shape)
    # print(processed_records.head())

    
    X_train = processed_records['vector_data']
    y_train = processed_records['queen_status']
    X_train_ok = np.array(X_train.tolist())
    return X_train_ok, y_train

def load_testing_data_for_knn(cfg):
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    processed_records = data_lib.load_and_prepare_data_for_vector_db(cfg, "test")
    # print(processed_records.shape)
    # print(processed_records.head())
    
    X_test = processed_records['vector_data']
    y_test = processed_records['queen_status']
    X_test_ok = np.array(X_test.tolist())
    return X_test_ok, y_test

def execute_and_test_knn(X_train, y_train, X_test, y_test, cfg):
    training_set_size = len(y_train)
    testing_set_size = len(y_test)
    neighbors = cfg._INDEX_PARAMS["params"].get("neighbors", 10)
    vote_type = cfg._INDEX_PARAMS["index_type"]
    print(f"neighbors:{neighbors};vote type={vote_type}")
    knn = KNeighborsClassifier(n_neighbors=neighbors,
                               metric=cfg._INDEX_PARAMS["metric_type"],
                               weights=vote_type)
    
    if (cfg._NORMALIZE_VECTOR):
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    print("Fit values:")
    print(X_train_scaled.shape)
    # print(X_train_scaled[:5])
    print("data type=", X_train_scaled.dtype)
    
    # start = time.perf_counter()
    # mem_start = lib.mem_KB()
    # knn.fit(X_train_scaled, y_train)
    # train_memory = lib.mem_KB() - mem_start
    # train_elapsed = time.perf_counter() - start
    metrics = lib.measure_perf(lambda: (knn.fit(X_train_scaled, y_train)))
    print("metrics=", metrics)
    train_elapsed = metrics["elapsed_time"]
    train_memory = metrics["peak_memory_MB"]
    
    print("Predict values:")
    print(X_test_scaled.shape)
    print(X_test_scaled[:5])
    # start = time.perf_counter()
    # mem_start = lib.mem_KB()
    # y_pred = knn.predict(X_test_scaled)
    # predict_memory = lib.mem_KB() - mem_start
    # predict_elapsed = time.perf_counter() - start
    metrics = lib.measure_perf(lambda: knn.predict(X_test_scaled))
    predict_memory = metrics["peak_memory_MB"]
    predict_elapsed = metrics["elapsed_time"]
    y_pred = metrics['result']
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Acurracy KNN: {accuracy * 100:.2f}%")
    metrics_json = {
        "neighbors": neighbors,
        "vote_type":vote_type,
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
    do_normalize = cfg._NORMALIZE_VECTOR
    cfg._NORMALIZE_VECTOR = False
    
    orginal_overlap = cfg._SEGMENT_OVERLAP
    if (cfg._SEGMENT_OVERLAP == 'all'):
        overlaps = GlobalVars.get_all_overlaps_available(segment_lenght=cfg._SEGMENT_LENGHT)
    else: 
        overlaps = [cfg._SEGMENT_OVERLAP]
    
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []

    for overlap in overlaps:
        cfg._SEGMENT_OVERLAP = overlap
        X_train, y_train = load_training_data_for_knn(cfg)
        X_test, y_test = load_testing_data_for_knn(cfg)
        
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    all_X_train = np.concatenate(all_X_train, axis=0)      # pentru ndarray
    all_y_train = pd.concat(all_y_train, ignore_index=True)
    all_X_test = np.concatenate(all_X_test, axis=0)      # pentru ndarray
    all_y_test = pd.concat(all_y_test, ignore_index=True)
    
    cfg._SEGMENT_OVERLAP = orginal_overlap
        
    
    # X_train, y_train = load_data_for_knn(cfg)
    # X_test, y_test = load_testing_data_for_knn(cfg)
    cfg._NORMALIZE_VECTOR = do_normalize
    metrics_json, y_pred = execute_and_test_knn(all_X_train, all_y_train, all_X_test, all_y_test, cfg)
    return metrics_json

def execute_configuration_as_process(configuration, save_results = 1, display_std_out = True):
    if (type(configuration) == 'dict' or isinstance(configuration, dict)):
        params_str = json.dumps(configuration) 
    else:
        params_str = configuration
    
    #print("params as json=", params_str)
    
    md5_hash = hashlib.md5()
    md5_hash.update(params_str.encode('utf-8'))
    process_key = md5_hash.hexdigest()
    errors = None
    
    process_parameters = ["python", "knn_experiment.py", process_key, str(save_results), params_str]
    #print("prpcess key=", process_key)
    process = subprocess.Popen(process_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()  # wait for finish 
    if (stdout != '' and display_std_out == True):
        print("stdout:", stdout)
        
    knn_results = None
    error = None
    if (stderr == ''):        
        knn_results_path = GlobalVars.knn_results_path + f"{process_key}.knn"
        with open(knn_results_path, 'r') as json_file:
            knn_results = json.load(json_file)
        #print("Result:", errors)
    else:
        print("Erori:", stderr)
        # logging.error(f"Process key: {process_key} with configs:\n{params_str}")
        # logging.error(stderr)
        error = stderr
    
    return process_key, knn_results, error

def report_results(results_file_path, top_n):
    print(" ########## Reports ##############")
    df_sorted = lib.load_resuls_file(results_file_path)
    df_top_n = df_sorted.head(top_n)

    df_top_n = df_top_n.drop(['skipped', 'training_set_size', 'testing_set_size'], axis=1)
    df_top_n_copy = df_top_n.copy()
    df_top_n_copy = df_top_n_copy.drop(['row_key'], axis=1)    
    labels_desc = lib.get_queen_status_description_map()
    labels = [labels_desc[0], labels_desc[1], labels_desc[2], labels_desc[3]]

    prev_acurracy_score = 0
    for index, row in df_top_n.iterrows():
        process_key = row['row_key']    
        acurracy_score = row['accuracy']
        if (acurracy_score == prev_acurracy_score):
            continue
        prev_acurracy_score = acurracy_score
        knn_results_path = GlobalVars.knn_results_path + f"{process_key}.knn"
        with open(knn_results_path, 'r') as json_file:
            faiss_results = json.load(json_file)
            
        title = f"Config: segment({row['segment_lenght']}, {row['segment_overlap']}), feat({row['feature']}, {row['vector_operation']}), metric({row['metric_type']},{row['index_params']}, {row['index_type']}) \nAccuracy {faiss_results['accuracy_score']*100:.2f}\n"
        print(title)
        cm = np.array(faiss_results['confusion_matrix'])
        classification_report = faiss_results['classification_report']
        lib.display_confusion_matrix(cm)
        lib.display_classification_report(classification_report)
         
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        os.makedirs(GlobalVars.knn_results_path, exist_ok=True)
        process_key = sys.argv[1]
        save_results = int(sys.argv[2])        
        configs_json = json.loads(sys.argv[3])  #json decode
        print("configs json=", configs_json)
        random_seed = configs_json.get("random_seed", 0)
        print(f"Param random seed: {random_seed}")
        
        lib.init_random_seed(random_seed)
        
        cfg = ExperimentConfig(configs_json)
        GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    
        knn_results = execute_configuration(cfg)
        print(knn_results)
        #logging.info(f"Process key: {process_key} with configs:\n{sys.argv[3]}")
        
        # if (save_results):        
        #     predictions_path = f"predictions{os.sep}{process_key}.csv"
        #     df_pred = pd.DataFrame({'y_test': y_test.flatten(), 'y_pred': y_pred.flatten()})
        #     df_pred.to_csv(predictions_path, index=True) 
            
        #errors save always
        knn_results_path = GlobalVars.knn_results_path +f"{process_key}.knn"
        with open(knn_results_path, 'w') as json_file:
            json.dump(knn_results, json_file, indent=4)
        exit(0)
    else:
        print("No params!")
        exit(1)