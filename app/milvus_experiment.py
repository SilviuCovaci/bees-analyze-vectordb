import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import json
import os
from itertools import product

import data_library as data_lib
import library as lib
from library import GlobalVars
from library import ExperimentConfig

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from  helpers.milvus_helper import MilvusHelper
import helpers.dask_helper as dask_helper
import data_library as data_library
import time


debug_mode = True
helper = None
client = None
batch_size = 100


def combine_multiple_parameters():
    segment_prop = [(2,0), (2,1), (5,0), (5,2), (10, 0), (10,5)]
    is_balanced = [1, 2]
    features = ['mfcc_20', 'mfcc_30', 'mfcc_40', 'mfcc_50']
    operations = ['mean', 'mean_iqr15', 'mean_iqr']
    metric_type = ['L2', 'COSINE']
    index_params = [
        {'index_type': 'FLAT', 'params': {}},
        {'index_type': 'IVF_FLAT', 'params': {'nlist': 64}}, {'index_type': 'IVF_FLAT', 'params': {'nlist': 128}}, {'index_type': 'IVF_FLAT', 'params': {'nlist': 256}},
        {'index_type': 'IVF_PQ', 'params': {'nlist': 64, 'm': 1, 'nbits': 8}},{'index_type': 'IVF_PQ', 'params': {'nlist': 64, 'm': 10, 'nbits': 8}},
        {'index_type': 'IVF_PQ', 'params': {'nlist': 128, 'm': 1, 'nbits': 16}}, {'index_type': 'IVF_PQ', 'params': {'nlist': 128, 'm': 10, 'nbits': 16}},
        {'index_type': 'IVF_SQ8', 'params': {'nlist': 64, 'm': 1, 'nbits': 8}}, {'index_type': 'IVF_SQ8', 'params': {'nlist': 64, 'm': 10, 'nbits': 8}},
        {'index_type': 'IVF_SQ8', 'params': {'nlist': 128, 'm': 1, 'nbits': 16}}, {'index_type': 'IVF_SQ8', 'params': {'nlist': 128, 'm': 8, 'nbits': 16}}, {'index_type': 'IVF_SQ8', 'params': {'nlist': 128, 'm': 16, 'nbits': 16}},        
        {'index_type': 'HNSW', 'params': {'M': 16, 'efConstruction': 200}}, {'index_type': 'HNSW', 'params': {'M': 32, 'efConstruction': 200}}, {'index_type': 'HNSW', 'params': {'M': 64, 'efConstruction': 200}},
        {'index_type': 'HNSW', 'params': {'M': 16, 'efConstruction': 400}}, {'index_type': 'HNSW', 'params': {'M': 32, 'efConstruction': 400}}, {'index_type': 'HNSW', 'params': {'M': 64, 'efConstruction': 400}},
    ]

    #all posible combinations
    param_combinations = list(product(
        segment_prop, is_balanced, features, operations, metric_type, index_params 
    ))

    all_configurations = [
        {
            "segment_lenght": segp[0],            
            "segment_overlap": segp[1],    
            "balanced_type": bal,       
            "feature": feat,
            "vector_operation": operation,     
            "metric_type": m_type,
            "index_type": idx_data['index_type'],
            "index_params": idx_data['params']
        
        }
        for segp, bal, feat, operation, m_type, idx_data in param_combinations
    ]
    df_configuration = pd.DataFrame(all_configurations)
    df_configuration.to_csv('experiments' + os.sep + 'all_experiments.csv', index=False) 
    

    
def init_milvus():
    global helper
    global client
    
    if (helper is None):
        helper = MilvusHelper(debug_mode)
    
    if (client is None):
        client = helper.get_client()
    
def set_milvus(intialised_helper, initialised_client):
    global helper
    global client
    
    helper = intialised_helper
    client = initialised_client
    
def get_or_create_collection_for_config(cfg, recreate_if_exists = False, reset_if_exists = False, load_collection = False):   
    collection_name = cfg._COLLECTION_NAME
    helper.debug("collection_name=", collection_name)
    collection = helper.get_collection(collection_name=collection_name)
    
    #print("collection=", collection)
    if (not collection is None):
        if (recreate_if_exists):
            drop_collection(collection_name)            
            collection = None
            time.sleep(3)
        if (reset_if_exists):
            collection = helper.clear_collection(collection_name=cfg._COLLECTION_NAME)
            
    if (collection is None):
        helper.create_vector_collection(collection_name=collection_name, field_name=cfg._FIELD_NAME, vector_dimension=cfg._FIELD_DIM, index_params=cfg._INDEX_PARAMS )
        collection = helper.get_collection(collection_name)
        helper.debug("Vector dim=", cfg._FIELD_DIM, collection.num_entities)
    
    if (collection.has_index()):
        index = collection.index()
        helper.debug("INDEX INFO:", index.params)
        if (load_collection):
            helper.load_collection_if_need(collection_name=cfg._COLLECTION_NAME)
    else:
        helper.debug("NO INDEX INFO:")
        
    return collection, collection_name

def drop_collection(collection_name):
    client.drop_collection(
        collection_name=collection_name
    )  
    
    


 
    
def add_data_in_vector_db(processed_records, helper, collection_name, field_name, batch_size=100):
    """
    Add data in specified collection in a sequential mode

    Args:
        processed_records (_type_): _description_
        helper (_type_): _description_
        collection_name (_type_): _description_
        field_name (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 100.
    """
    
    batch_records = []
    for _, record in tqdm( processed_records.iterrows(), total=processed_records.shape[0]):
        record[field_name] = list(map(float, record[field_name]))
        batch_records.append(record.to_dict())
        if len(batch_records) >= batch_size:
            helper.insert_records(collection_name, batch_records)
            batch_records = []  
            
    if batch_records and len(batch_records) > 0:
        helper.insert_records(collection_name, batch_records)
    
def process_and_insert_batch(batch, cfg):      
          
    batch_records = []
    for record in batch:
        batch_records.append(record)
        if len(batch_records) >= batch_size:
            helper.insert_records(collection_name = cfg._COLLECTION_NAME, records = batch_records, do_flush = False)
            batch_records = []  
            
    if batch_records and len(batch_records) > 0:
        helper.insert_records(collection_name = cfg._COLLECTION_NAME, records = batch_records, do_flush = False)
                
   

def load_train_data_for_milvus(cfg):
    orginal_overlap = cfg._SEGMENT_OVERLAP
    if (cfg._SEGMENT_OVERLAP == 'all'):
        overlaps = GlobalVars.get_all_overlaps_available(segment_lenght=cfg._SEGMENT_LENGHT)
    else: 
        overlaps = [cfg._SEGMENT_OVERLAP]
        
    all_dfs = []
    
    for overlap in overlaps:
        cfg._SEGMENT_OVERLAP = overlap
        processed_records = data_lib.load_and_prepare_data_for_vector_db(cfg, "train")
        all_dfs.append(processed_records)
    
    all_processed_records = pd.concat(all_dfs, ignore_index=True)        
    return all_processed_records
    
def create_and_fill_collection_for_specified_configuration(cfg, recreate_if_exists = False):
    print("test1")
    init_milvus()
    print("Initialised OK!")
    collection, collection_name = get_or_create_collection_for_config(cfg, recreate_if_exists=recreate_if_exists)    
    print(f"Check collection: {collection_name}; recreated: {recreate_if_exists}")
    
    processed_records = load_train_data_for_milvus(cfg)
    if (cfg._NORMALIZE_VECTOR):
        X_test = processed_records['vector_data']
        X_test_ok = np.array(X_test.tolist())
    
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        X_train_scaled = scaler.fit_transform(X_test_ok)
        processed_records['vector_data'] = list(X_train_scaled)
    
    print(processed_records.shape)
    print(processed_records.head(10))

    record = processed_records.iloc[1]
    print(record)
    print(record[cfg._FIELD_NAME].shape)
    
    """
    Add data in collection in a paralel way (using dask)
    """
    dask_helper.process_records_in_paralel(processed_records, lambda batch: process_and_insert_batch(batch, cfg), dask_scheduler='threads') 
    collection.flush()
    print("Collection flushed!!!")
    helper.recreate_vector_index(collection_obj=collection, field_name=cfg._FIELD_NAME, index_params=cfg._INDEX_PARAMS )
    
    return processed_records.size, collection_name
    
def load_collection(cfg):
    #collection_name = GlobalVars.get_collection_name_for_experiment(cfg)
    collection_name = cfg._COLLECTION_NAME
    helper.debug(f"collection_name={collection_name}")
    collection = helper.get_collection(collection_name)
    helper.debug("collection=", collection)
    if (collection is None):
        print(f"COLLECTION {cfg._COLLECTION_NAME} FOR SEARCHING DATA NOT FOUND!")
    else:
        print(collection.num_entities, " items")        
    try:
        index = collection.index()
        print("Informații detaliate despre index actual :", type(index.params), index.params)
    except Exception as e:
        print(f"Nu am reusit sa obtin indexul: {e}")
    
    return collection, collection_name

def extract_parts_from_sub_segment_name(input_string):
    # Separăm string-ul după caracterul '_'
    parts = input_string.split('_')
    
    segment_name = '_'.join(parts[:4])    
    segment_set_name = '_'.join(parts[:2])    
    
    return segment_name, segment_set_name
    
    
def predict_with_milvus(milvus_index, test_data, cfg):
    
    neighbors = cfg._INDEX_PARAMS["params"].get("neighbors", None)     
    search_params = {}
    if "nprobe" in cfg._INDEX_PARAMS["params"]:
        nprobe = cfg._INDEX_PARAMS["params"].get("nprobe", None)            
        search_params['nprobe'] = nprobe
    
    
    if "efSearch" in cfg._INDEX_PARAMS["params"]:
        efSearch = cfg._INDEX_PARAMS["params"].get("efSearch", None)  
        search_params['efSearch'] = efSearch
        
    helper.debug(f"apply milvus search for neighbors {neighbors} and {search_params}")
        
    batch_size = 10000  # sau 10000, dar sub 16384
    results = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batch_results = milvus_index.search(
                collection_name=cfg._COLLECTION_NAME,
                anns_field= cfg._FIELD_NAME,
                data=batch,
                limit=neighbors, 
                search_params={"params": search_params},
                output_fields=["queen_status"],
            )
        results.extend(batch_results)
    
    #print("len test data=", len(test_data), "results=", len(results))
    neighbor_labels = [[hit.get('entity').get("queen_status", None) for hit in hits] for hits in results]
    predictions = np.apply_along_axis(lib.majority_vote, axis=1, arr=np.array(neighbor_labels))

    return predictions.tolist()

def predict_one_row(csv_record, cfg):
    final_vector = lib.load_and_prepare_features_for_vectordb(csv_record, cfg)        
    res = client.search(
            collection_name=cfg._COLLECTION_NAME,
            anns_field= cfg._FIELD_NAME,
            data=[final_vector],
            limit=1,  # Ajustează în funcție de numărul de rezultate dorite
            search_params={"metric_type": cfg._INDEX_PARAMS["metric_type"], "nprobe": 32},
            output_fields=["sub_segment_name", "queen_status"],
        )
    
    predicted_status = None
    hit_sub_segment_name = None
    if res:
        for hits in res:
            for idx, hit in enumerate(hits):  
                predicted_status = hit['entity'].get("queen_status", None)
                hit_sub_segment_name = hit['entity'].get("sub_segment_name", None)

                        
    prediction = {
            'sub_segment_name': csv_record['sub_segment_name'],
            'queen_status': csv_record['queen_status'],
            'predicted_status': predicted_status,
            'predicted_sub_segment_name': hit_sub_segment_name
    }                        
    return pd.Series(prediction)

# def predict_batch_rows(batch, cfg):      
#     predicted_rows = [predict_one_row(record, cfg) for record in batch]
#     return pd.DataFrame.from_records(predicted_rows)

        
def do_prediction_distributed(test_data, cfg, save_results = None):
    if (helper is None):
        init_milvus()
    helper.load_collection_if_need(collection_name=cfg._COLLECTION_NAME)
    helper.describe_index(collection_name=cfg._COLLECTION_NAME)
    meta = {
        'sub_segment_name': 'object',         
        'queen_status': 'int8',  # Corectare cheie
        'predicted_status': 'int8', 
        'predicted_sub_segment_name': 'object', 
    }
    
    processed_records = lib.load_and_prepare_data_for_vector_db(cfg, "test")
    predicted_records = dask_helper.process_dataframe(processed_records, lambda csv_record: predict_one_row(csv_record, cfg),  meta=meta, dask_scheduler='threads') 
    #print("predicted records=", predicted_records)
    #predicted_records = dask_helper.process_csv(testing_file_path, lambda csv_record: predict_one_row(csv_record, cfg), blocksize='200KB', meta=meta, dask_scheduler='threads')
    if (save_results is not None):        
        predicted_records.to_csv(f"results/{save_results}.csv", index=False)
        print("CSV salvat cu succes!")
    
    accuracy = accuracy_score(predicted_records["queen_status"], predicted_records["predicted_status"])
    conf_matrix = confusion_matrix(predicted_records["queen_status"], predicted_records["predicted_status"]).tolist()
    class_report = classification_report(predicted_records["queen_status"], predicted_records["predicted_status"], output_dict=True)

    metrics = {
        "accuracy_score": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }
    metrics_json = json.dumps(metrics, indent=4)
    return metrics, metrics_json, predicted_records.size
    

def do_prediction(testing_data, collection_name, cfg, save_results = None):
    
    testing_data = testing_data.head(1000)
    helper.load_collection_if_need(collection_name=collection_name)
    
    # Lista pentru a salva rezultatele de precizie
    predictions = []
    correct_predictions = 0
    first_correct_predictions = 0
    same_segments_no = 0
    same_segments_set_no = 0
    #total_records = 1000  # Luăm primele 100 de înregistrări
    total_records = testing_data.shape[0]
    
    isolated_testing_items = 0
    isolated_correct_predictions = 0
    # Iterează prin primele 100 de înregistrări din validation_data
    #for idx in range(total_records):
    for _, csv_record in tqdm( testing_data.iterrows(), total=testing_data.shape[0]):
        hit_segment_name = ''
        hit_segments_set_name = ''
        test_segment_name, test_segments_set_name = extract_parts_from_sub_segment_name(csv_record['sub_segment_name'])
        final_vector = load_and_prepare_features_for_vectordb(csv_record, cfg)  
        # Execută căutarea pentru vectorul curent
        res = client.search(
            collection_name=collection_name,
            anns_field= cfg._FIELD_NAME,
            data=[final_vector],
            limit=5,  # Ajustează în funcție de numărul de rezultate dorite
            search_params={"metric_type": cfg._INDEX_PARAMS["metric_type"]},
            output_fields=["sub_segment_name", "queen_status"]
        )
        # Extrage valoarea de queen_status predicționată din rezultate
        status_scores = {}
        predicted_status = None
        first_predicted_status = None
        # Procesăm fiecare rezultat
        if res:
            for hits in res:
                for idx, hit in enumerate(hits):  # Enumerăm pentru a lua poziția
                    predicted_status = hit['entity'].get("queen_status", None)
                    hit_sub_segment_name = hit['entity'].get("sub_segment_name", None)
                    hit_segment_name, hit_segments_set_name = extract_parts_from_sub_segment_name(hit_sub_segment_name)
                    if (first_predicted_status is None):
                        first_predicted_status = predicted_status
                        
                    if predicted_status:
                        # Atribuim scorul în funcție de poziție și de apariții
                        score = 1 / (idx + 1)  # Scor mai mare pentru poziții mai bune (ex: 1, 0.5, 0.33,...)
                        
                        if predicted_status in status_scores:
                            status_scores[predicted_status] += score  # Adăugăm scorul dacă deja există
                        else:
                            status_scores[predicted_status] = score  # Adăugăm un scor nou
        
        # Identifică statusul cu cel mai mare scor
        predicted_status = max(status_scores, key=status_scores.get, default=None)

        # Compară queen_status-ul real cu cel prezis
        if predicted_status is not None and predicted_status == csv_record['queen status']:
            correct_predictions += 1
        
        if (hit_segments_set_name != test_segments_set_name):
            isolated_testing_items +=1
            
        if (first_predicted_status is not None and first_predicted_status == csv_record['queen status']):
            first_correct_predictions += 1
            if (hit_segment_name == test_segment_name):
                same_segments_no += 1
            if (hit_segments_set_name == test_segments_set_name):
                same_segments_set_no += 1

            if (hit_segments_set_name != test_segments_set_name):                
                isolated_correct_predictions += 1
                
        prediction = {
            'sub_segment_name': csv_record['sub_segment_name'],
            'queen_status': csv_record['queen status'],
            'predicted_status': first_predicted_status,
            'predicted_sub_segment_name': hit_sub_segment_name
        }
        predictions.append(prediction)
        
                
    # Calcularea acurateței
    accuracy = correct_predictions / total_records * 100
    print(f"Accuracy: {accuracy}%")

    accuracy2 = first_correct_predictions / total_records * 100
    print(f"Accuracy2: {accuracy2}%, hits found {first_correct_predictions}")
    
    same_segment_hit = first_correct_predictions / total_records * 100
    print(f"Hits from same segment fle name: {same_segments_no} from {first_correct_predictions} hits")
    print(f"Hits from same set of segments: {same_segments_set_no} from {first_correct_predictions} hits")
    
    accuracy3 = isolated_correct_predictions / isolated_testing_items * 100
    print(f"Isolated items accuracy: {accuracy3}%, hits found {isolated_correct_predictions} from {isolated_testing_items}")
    
    results_df = pd.DataFrame(predictions)
    if (save_results is not None):        
        results_df.to_csv(f"results/{save_results}.csv", index=False)
        print("CSV salvat cu succes!")
    
    accuracyscore = accuracy_score(results_df["queen_status"], results_df["predicted_status"])
    conf_matrix = confusion_matrix(results_df["queen_status"], results_df["predicted_status"]).tolist()
    class_report = classification_report(results_df["queen_status"], results_df["predicted_status"], output_dict=True)

    metrics_json = json.dumps({
        'accuracy': accuracy,
        'accuracy2': accuracy2,
        "accuracy_score": accuracyscore,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }, indent=4)
    return metrics_json, total_records
    
def execute_experiment(cfg_json, save_results = None):
    cfg = ExperimentConfig(cfg_json)
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    print(cfg)

    #if (helper is None):
    init_milvus()
    training_set_size, collection_name = create_and_fill_collection_for_specified_configuration(cfg)
    time.sleep(1) 
    _, testing_path = lib.build_testing_data_from_extended_data(
                        cfg._BALANCED_TYPE, 
                        include_regular_segments=cfg._USE_REGULAR_SEGMENTS, 
                        include_first_segment=cfg._USE_FIRST_SEGMENT, 
                        include_last_segment=cfg._USE_LAST_SEGMENT)
    results, results_json, tested_items_no = do_prediction_distributed(testing_path, cfg, save_results)
    drop_collection(collection_name)
    return results, results_json, training_set_size, testing_set_size        


def create_and_fill_all_collections(do_normalize = False, balanced_type = 1, default_metric_type="L2", default_index_type="FLAT", default_index_params=""):
    results_file_path='experiments' + os.sep + 'all_collections.csv'
    df=pd.read_csv(results_file_path)
    
    start_index = 0
    end_index = len(df)
    print(f"end index={end_index}")
    for row in tqdm(df.iloc[start_index:end_index].itertuples(index=True), total=end_index - start_index):
        cfg_dict = cfg_dict = row._asdict()
        cfg_dict["normalize"] = do_normalize
        cfg_dict["balanced_type"] = balanced_type
        cfg_dict["features"] = [cfg_dict['feature']]
        cfg_dict["field_dim"] = ExperimentConfig.get_field_dim_by_feature(cfg_dict['feature'])
        cfg_dict["metric_type"]= default_metric_type
        cfg_dict["index_type"] = default_index_type
        cfg_dict["index_params"] = default_index_params

        cfg = ExperimentConfig(cfg_dict)
        GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
        create_and_fill_collection_for_specified_configuration(cfg)
    
    print("Done!")

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        process_key = sys.argv[1]
        save_results = int(sys.argv[2])        
        configs_json = json.loads(sys.argv[3])  #json decode        
        print(configs_json)
        results, results_json, training_set_size, testing_set_size = execute_experiment(configs_json, save_results=process_key if save_results else None)
        
        #logging.info(f"Process key: {process_key} with configs:\n{sys.argv[3]}")
    
        result_path = f"metrics{os.sep}{process_key}.json"
        with open(result_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        exit(0)
    else:
        print("No params!")
        exit(1)
        

# def recreate_index(collection, field_name, index_params):
#     print("recreate index with params:", index_params)
#     helper.recreate_vector_index(collection_obj=collection, field_name=field_name, index_params=index_params)
#     print("compact collection!")
#     collection.compact()
#     index = collection.index()
#     print("Informații detaliate despre index recreat:", index.params)
    
def recreate_index(collection_obj, cfg, wait_till_ready=200):
    return helper.recreate_vector_index(collection_obj=collection_obj, field_name=cfg._FIELD_NAME, index_params=cfg._INDEX_PARAMS, wait_till_ready=wait_till_ready)
    
"""


"""
def execute_and_test_milvus(X_train, y_train, X_test, y_test, cfg):
    training_set_size = len(y_train)
    testing_set_size = len(y_test)
    
    if (cfg._NORMALIZE_VECTOR):
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        
    init_milvus()
    collection, collection_name = get_or_create_collection_for_config(cfg, recreate_if_exists=False, load_collection=True)    

    metrics = lib.measure_perf(recreate_index, collection, cfg)
    has_index, create_index_elapsed_time = metrics['result'];
    helper.debug("recreate index result=", )
    train_elapsed = metrics["elapsed_time"]
    train_memory = metrics["peak_memory_MB"]
    if (has_index):
        metrics["elapsed_time"] = create_index_elapsed_time
    
    helper.load_collection_if_need(collection_name)
    
    helper.debug("Call predict!")
    metrics = lib.measure_perf(
        predict_with_milvus,
        milvus_index = client,
        test_data=X_test_scaled,
        cfg=cfg        
    )
    predict_elapsed = metrics["elapsed_time"]
    predict_memory = metrics["peak_memory_MB"]
    y_pred = metrics['result']
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"Acurracy MILVUS: {accuracy * 100:.2f}%")
    neighbors = cfg._INDEX_PARAMS["params"].get("neighbors", None)     
    vote_type = 'uniform'
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


def execute_configuration(cfg, debug_mode = False):

    all_X_train, all_y_train, all_X_test, all_y_test = data_lib.load_training_and_testing_data_for_vectordb(cfg)
    #dim = all_X_train.shape[1]
    #print("execute_configuration train dim=", dim)
    metrics_json, y_pred = execute_and_test_milvus(all_X_train, all_y_train, all_X_test, all_y_test, cfg)
    return metrics_json