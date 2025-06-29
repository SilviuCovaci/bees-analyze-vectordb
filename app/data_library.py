import pandas as pd
import numpy as np
import os
from itertools import product
from tqdm import tqdm
import pickle

from sklearn.preprocessing import MinMaxScaler

import library as lib
from library import GlobalVars
from library import ExperimentConfig
import helpers.dask_helper as dask_helper


def combine_all_data_parameters():
    #segment_prop = [(2,0), (2,1), (5,0), (5,2), (7,5), (10, 0), (10,5)]
    segment_prop = [(5,0), (5,2), (7,5), (10, 0), (10,5)]
    #features = ['mfcc_13', 'mfcc_20', 'mfcc_30', 'mfcc_40', 'mfcc_50']
    features = ['pe-mfcc_20', 'pe-mfcc_30', 'pe-mfcc_40', 'pe-mfcc_50']
    operations = ['mean', 'mean_iqr15', 'mean_iqr25']
       
    
    #all posible combinations
    param_combinations = list(product(
        segment_prop, features, operations
    ))

    all_configurations = [
        {
            "segment_lenght": segp[0],            
            "segment_overlap": segp[1],         
            "feature": feat,
            "vector_operation": operation
        
        }
        for segp, feat, operation in param_combinations
    ]
    df_configuration = pd.DataFrame(all_configurations)
    
    os.makedirs(GlobalVars.experiments_path, exist_ok=True)
    df_configuration.to_csv(GlobalVars.experiments_path + 'all_collections.csv', index=False) 


def load_and_prepare_features_for_vectordb(csv_record, cfg):
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    all_vectors = []
    for feature_name in cfg._FEATURES:
        parts = feature_name.split('+', 1)  # împarte o singură dată

        if len(parts) == 1:
            parts.append('')

        feature_name = parts[0]
        feature_option = parts[1]
        feature_vector = lib.load_one_feature_from_db(feature_name, csv_record['sub_segment_name'])
        processed_vector = lib.prepare_vector_for_db(vector = feature_vector, vector_operation = cfg._VECTOR_OPERATION)
        if (feature_option == 'stat'):
            mean = np.mean(processed_vector)
            std = np.std(processed_vector)
            processed_vector = np.append(processed_vector, [mean, std])
        all_vectors.append(processed_vector)
        
    final_vector = np.concatenate(all_vectors)            
    return final_vector

def process_one_row_for_import(csv_record, cfg):
    
    field_name = cfg._FIELD_NAME
    final_vector = load_and_prepare_features_for_vectordb(csv_record, cfg)
    
        
    record = {
             field_name: final_vector, 
             'queen_status': csv_record['queen status'], 
             'queen_presence': csv_record['queen presence'], 
             'queen_acceptance': csv_record['queen acceptance'], 
             'segment_no': csv_record['segment_no'], 
             'sub_segment_name': csv_record['sub_segment_name'],  
             'sub_segment_no': csv_record['sub_segment_no'],  
             'file_name_prefix': csv_record['file_name_prefix'],  
             'train1': csv_record['train1'],  
             'train2': csv_record['train2'],  
    }
    return pd.Series(record)

def load_and_prepare_data_for_vector_db(cfg, train_or_test = 'train', options = {}):
    use_cache=options.get('use_cache', True)
    include_first_segment=options.get('include_first_segment', False)
    include_last_segment=options.get('include_last_segment', False)
    include_regular_segments=options.get('include_regular_segments', True)

    cache_file_path = GlobalVars.get_cache_path_for_experiment(cfg)
    print(f"{train_or_test}:{cache_file_path}")
    if ((use_cache == True) and os.path.exists(cache_file_path)):
        processed_records = load_cache(cache_file_path)

        # processed_records = pd.read_csv(cache_file_path)
        # processed_records["vector_data"] = processed_records["vector_data"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    else:
        field_name = cfg._FIELD_NAME
        csv_file_path = GlobalVars.extended_dataset_file_path(segment_length = cfg._SEGMENT_LENGHT, overlap = cfg._SEGMENT_OVERLAP)
        # if (train_or_test == 'train'):
        #     _, csv_file_path = lib.build_training_data_from_extended_data(cfg._BALANCED_TYPE)
        # else:
        #     _, csv_file_path = lib.build_testing_data_from_extended_data(
        #                         cfg._BALANCED_TYPE, 
        #                         include_regular_segments=cfg._USE_REGULAR_SEGMENTS, 
        #                         include_first_segment=cfg._USE_FIRST_SEGMENT, 
        #                         include_last_segment=cfg._USE_LAST_SEGMENT
                            # )
        dtype={'queen status': 'int8', 'queen presence': 'int8', 'queen acceptance': 'int8', 'segment_no': 'int8', 'sub_segment_name': 'object', 'sub_segment_no': 'int8'}
        dtype={'file_name_prefix': 'object', 
               'segment_file': 'object', 
               'queen presence': 'int8',  
               'queen acceptance': 'int8', 
               'queen status': 'int8', 
               'segment_no': 'int8', 
               'sub_segment_name': 'object', 
               'sub_segment_no': 'int8',
               'start_time': 'int8',
               'end_time': 'int8',
               'train1': 'int8',
               'train2': 'int8'}
        meta = {
            field_name: 'object',         
            'queen_status': 'int8',  # Corectare cheie
            'queen_presence': 'int8', 
            'queen_acceptance': 'int8', 
            'segment_no': 'int8',
            'sub_segment_name': 'object', 
            'sub_segment_no': 'int8',
            'file_name_prefix': 'object',
            'train1': 'int8',
            'train2': 'int8'
        }
        
        processed_records = dask_helper.process_csv(csv_file_path, lambda csv_record: process_one_row_for_import(csv_record, cfg), blocksize='200KB', meta=meta, dtype=dtype)                         
        processed_records["vector_mean"] = processed_records["vector_data"].apply(lambda v: np.mean(v))
        processed_records["vector_std"] = processed_records["vector_data"].apply(lambda v: np.std(v))
        # print(processed_records.head(10))
        if (use_cache == True):
            save_cache(processed_records, cache_file_path)

    if (train_or_test == 'train'):
        processed_records = processed_records[processed_records[f'train{cfg._BALANCED_TYPE}'] == 1]
    elif (train_or_test == 'test'):
        processed_records = processed_records[processed_records[f'train{cfg._BALANCED_TYPE}'] == 0]
        
    conditions = []

    if include_first_segment:
        conditions.append(processed_records['segment_no'] == 0)
    
    if include_last_segment:
        conditions.append(processed_records['segment_no'] == 5)
    
    if include_regular_segments:
        conditions.append(processed_records['segment_no'].isin([1, 2, 3, 4]))

    if conditions:
        processed_records = processed_records[pd.concat(conditions, axis=1).any(axis=1)] 
    processed_records = processed_records.copy()
    processed_records.reset_index(drop=True, inplace=True)
    return processed_records

def cache_one_configuration(cfg):
    features = '_'.join(cfg._FEATURES)
    cache_file_path = GlobalVars.get_cache_path_for_experiment(cfg)
    print(f"Generate file:{cache_file_path}")
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    load_and_prepare_data_for_vector_db(cfg=cfg, train_or_test='')
    #load_and_prepare_data_for_vector_db(cfg=cfg, train_or_test='test', use_cache=True)    

def cache_all_collections(do_normalize = False, balanced_type = 1):
    results_file_path=GlobalVars.experiments_path + 'all_collections.csv'
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
        cfg = ExperimentConfig(cfg_dict)
        cache_one_configuration(cfg)
        
    
    print("Caching done!")
    
def load_cache(cache_file_path):
    with open(cache_file_path, "rb") as f:
        processed_records = pickle.load(f)
    return processed_records

def save_cache(processed_records, cache_file_path):
    with open(cache_file_path, "wb") as f:
        pickle.dump(processed_records, f)

def build_scaler_from_cache(scaler_name, margin = 0.1, balanced_type = 1):
	results_file_path='experiments' + os.sep + 'all_collections.csv'
	df=pd.read_csv(results_file_path)
	
	start_index = 0
	end_index = len(df)
	all_min_max = []
	
	for row in tqdm(df.iloc[start_index:end_index].itertuples(index=True), total=end_index - start_index):
		cfg_dict = cfg_dict = row._asdict()
		cfg_dict["normalize"] = False
		cfg_dict["balanced_type"] = balanced_type
		cfg_dict["features"] = [cfg_dict['feature']]
		cfg_dict["field_dim"] = ExperimentConfig.get_field_dim_by_feature(cfg_dict['feature'])
		cfg = ExperimentConfig(cfg_dict)
		
		features = '_'.join(cfg._FEATURES)
		cache_file_path = GlobalVars.get_cache_path_for_experiment(cfg)
		print(f"Load file:{cache_file_path}")
		GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
		train_records = load_and_prepare_data_for_vector_db(cfg=cfg, train_or_test='train', use_cache=True)
		test_records = load_and_prepare_data_for_vector_db(cfg=cfg, train_or_test='test', use_cache=True)
		
		train_vectors = np.vstack(train_records[cfg._FIELD_NAME])  
		all_min_max.append(np.min(train_vectors))
		all_min_max.append(np.max(train_vectors))
		
		test_vectors = np.vstack(test_records[cfg._FIELD_NAME])  
		all_min_max.append(np.min(test_vectors))
		all_min_max.append(np.max(test_vectors))
		
	print(all_min_max)
	min_vals = min(all_min_max)
	max_vals = max(all_min_max)
	range_vals = max_vals - min_vals
	min_vals = min_vals - margin * range_vals
	max_vals = max_vals + margin * range_vals
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(np.vstack([min_vals, max_vals])) 

	scaler_path = GlobalVars.dataset_path + scaler_name
	with open(scaler_path , "wb") as f:
		pickle.dump(scaler, f)
	print("Scaler saved")
     
    

def load_data_for_vectordb(cfg, data_type):
    GlobalVars.set_segment_lenght_and_overlap(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
    processed_records = load_and_prepare_data_for_vector_db(cfg, data_type)

    X = processed_records['vector_data']
    y = processed_records['queen_status']
    X_ok = np.array(X.tolist())
    return X_ok, y
     
def load_training_data_for_vectordb(cfg):
    return load_data_for_vectordb(cfg, "train")

def load_testing_data_for_vectordb(cfg):
    return load_data_for_vectordb(cfg, "test")

    
def load_training_and_testing_data_for_vectordb(cfg):
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
        X_train, y_train = load_training_data_for_vectordb(cfg)
        X_test, y_test = load_testing_data_for_vectordb(cfg)
    
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    all_X_train = np.concatenate(all_X_train, axis=0)      # for ndarray
    all_y_train = pd.concat(all_y_train, ignore_index=True)
    all_X_test = np.concatenate(all_X_test, axis=0)      # for ndarray
    all_y_test = pd.concat(all_y_test, ignore_index=True)
    
    cfg._SEGMENT_OVERLAP = orginal_overlap
    cfg._NORMALIZE_VECTOR = do_normalize
    return all_X_train, all_y_train, all_X_test, all_y_test

