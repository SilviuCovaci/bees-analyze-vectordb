import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import hashlib
import random

import sqlite3

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, io, shutil, json, ast, glob
from tqdm import tqdm 

from itertools import product
import psutil, time, threading

process = psutil.Process(os.getpid())

TESTING_PERCENT=0.2

CACHING_INDEX = True

class ExperimentConfig:
    def __init__(self, cfg_dict):
        one_feature = cfg_dict.get('feature', None)
        self._NORMALIZE_VECTOR = cfg_dict.get('normalize', True)        
        self._FEATURES = cfg_dict.get('features', None)
        if (self._FEATURES is None and (not one_feature is None)):
            self._FEATURES = [one_feature]
        self._ALL_CFG_FEATURES = '_'.join(self._FEATURES)    
            
        self._FIELD_DIM = cfg_dict.get('field_dim', None)
        if (self._FIELD_DIM is None) and (not self._FEATURES is None):
            self._FIELD_DIM = ExperimentConfig.get_field_dim_by_feature(self._ALL_CFG_FEATURES)
        self._FIELD_NAME = "vector_data"
        self._VECTOR_REDUCE_DIM = 20  # 64
        self._VECTOR_OPERATION = cfg_dict.get('vector_operation', "mean")
        
        self._BALANCED_TYPE = cfg_dict.get('balanced_type', 1)     
        index_specific_params = cfg_dict.get('index_params', '{}')
        
        if (not (type(index_specific_params) == 'dict' or isinstance(index_specific_params, dict))):
            index_specific_params = ast.literal_eval(index_specific_params)
            # if ("params" in index_specific_params):
            #     index_specific_params = index_specific_params.get("params", None)
        print("index_specific_params=", index_specific_params, type(index_specific_params))
        self._INDEX_PARAMS = {
            "metric_type": cfg_dict.get("metric_type", None),
            "index_type": cfg_dict.get("index_type", cfg_dict.get("vote_type", None)),            
            "params": index_specific_params
        }
        if ("neighbors" in cfg_dict):
            self._INDEX_PARAMS['params']['neighbors'] = cfg_dict.get("neighbors")
        self._SEARCH_PARAMS = cfg_dict.get('search_params', {})       
        self._SEGMENT_LENGHT = cfg_dict.get('segment_lenght', None)       
        self._SEGMENT_OVERLAP = cfg_dict.get('segment_overlap', None)   
        self._COLLECTION_NAME_PREFIX = f"{'_'.join(self._FEATURES)}_vectors"              
        
        self._USE_FIRST_SEGMENT = cfg_dict.get("include_first_segment", False)
        self._USE_LAST_SEGMENT = cfg_dict.get("include_last_segment", False)
        self._USE_REGULAR_SEGMENTS = cfg_dict.get("include_regular_segments", True)
        self._ROW_KEY = cfg_dict.get('row_key', "")       
        self._USE_CACHE = cfg_dict.get('use_cache', CACHING_INDEX)       
        self.update_collection_name()

    def update_collection_name(self, segment_length = None, overlap = None):
        if (not segment_length is None):
            self._SEGMENT_LENGHT = segment_length
        if (not overlap is None):
            self._SEGMENT_OVERLAP = overlap
        self._COLLECTION_NAME = f"{self._COLLECTION_NAME_PREFIX}_{self._VECTOR_OPERATION}_len{self._SEGMENT_LENGHT}_overlap{self._SEGMENT_OVERLAP}".replace("-", "_")
        
    def __str__(self):
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())

    @staticmethod
    def get_field_dim_by_feature(feature_name):
        if (feature_name == 'mfcc' or feature_name == 'mfcc_20' or feature_name == 'pe-mfcc_20'):
            field_dim = 20
        elif (feature_name == 'mfcc_30' or feature_name == 'pe-mfcc_30'):
            field_dim = 30
        elif (feature_name == 'mfcc_40' or feature_name == 'pe-mfcc_40'):
            field_dim = 40
        elif (feature_name == 'mfcc_50' or feature_name == 'pe-mfcc_50'):
            field_dim = 50
        elif (feature_name == 'mfcc_13' or feature_name == 'pe-mfcc_13'):
            field_dim = 13
        elif (feature_name == 'mfcc_40+stat' or feature_name == 'pe-mfcc_40+stat'):
            field_dim = 42
        else:
            print(f"Unknown dimension for feature: {feature_name}")
        return field_dim
    
class GlobalVars:
    current_date = pd.Timestamp.now()
    one_year_ago = current_date - pd.DateOffset(years=1)
    five_years_ago = (current_date - pd.DateOffset(years=5))
   
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    dataset_path = current_dir + os.sep + '..' +  os.sep + 'dataset' + os.sep
    csv_data_original = dataset_path + 'all_data_updated.csv'
    csv_data_processed_path = dataset_path + 'all_data_processed.csv'
    csv_data_sync_path = dataset_path + 'all_data_sync.csv'
    train_data_path = dataset_path + 'train_data.csv'
    test_data_path = dataset_path + 'test_data.csv'
    validate_data_path = dataset_path + 'validate_data.csv'
    
    experiments_path = current_dir + os.sep + '..' +  os.sep + 'experiments' + os.sep
    knn_results_path = current_dir + os.sep + '..' +  os.sep + 'knn_results' + os.sep
    faiss_results_path = current_dir + os.sep + '..' +  os.sep + 'faiss_results' + os.sep
    faiss_index_cache_path = current_dir + os.sep + '..' +  os.sep + 'faiss_index' + os.sep
    
    print("dataset_path=", dataset_path)
    soundfiles_path = dataset_path + 'sound_files' + os.sep
    mfcc_img_path = dataset_path + 'mfcc_img' + os.sep
    
    segment_length = None
    overlap = None
    suffix = 'undersampled'
    
    def set_segment_lenght_and_overlap(segment_length, overlap):
        GlobalVars.segment_length = segment_length
        GlobalVars.overlap = overlap
        res = os.makedirs(GlobalVars.extended_dataset_folder_path(), exist_ok=True)
        #print(f"Extended dataset folder={GlobalVars.extended_dataset_folder_path()}, res: {res}")
        
        
    def get_segment_lenght_and_overlap(segment_length = None, overlap = None):
        if (segment_length is None):
            segment_length = GlobalVars.segment_length
        
        if (overlap is None):
            overlap = GlobalVars.overlap
            
        return segment_length, overlap
    
    def create_folders_for_features(features):
        for feature in features:
            features_path = GlobalVars.features_path(feature_name = feature)        
            os.makedirs(features_path, exist_ok=True)
                    
    def features_path(segment_length = None, overlap = None, feature_name = None):
        return GlobalVars.extended_dataset_folder_path(segment_length, overlap=overlap) + f"{feature_name}_features"
    
    def feature_db_path(segment_length = None, overlap = None, feature_name = None, partition_id = None):
        if (partition_id is None):
            db_path = GlobalVars.features_path(segment_length=segment_length, overlap=overlap, feature_name=feature_name)  + ".db"
        else:
            db_path = GlobalVars.features_path(segment_length=segment_length, overlap=overlap, feature_name=feature_name)  + f"_p{partition_id}.db"
        return db_path
    
        
    def mel_features_path(segment_length = None, overlap = None):
        return GlobalVars.features_path(segment_length = segment_length, overlap = overlap, feature_name = 'mel')        
    
    def mfcc_features_path(segment_length = None, overlap = None):
        return GlobalVars.features_path(segment_length = segment_length, overlap = overlap, feature_name = 'mfcc')        
    
    def stft_features_path(segment_length = None, overlap = None):
        return GlobalVars.features_path(segment_length = segment_length, overlap = overlap, feature_name = 'stft')        
    
    
    # def mel_file_path(file_name_prefix, segment = 0, start_time = default_audio_start, duration = default_audio_duration):
    #     return GlobalVars.mel_features_path(start_time, duration) +  f'{file_name_prefix}__segment{segment}.npz'
    
    # def mfcc_file_path(file_name_prefix, segment = 0, start_time = default_audio_start, duration = default_audio_duration):
    #     return GlobalVars.mel_features_path(start_time, duration) +  f'{file_name_prefix}__segment{segment}.npz'
    
    # def wav_file_path(file_name_prefix, segment):
    #     return GlobalVars.soundfiles_path +  f'{file_name_prefix}__segment{segment}.wav'
    
    def extended_dataset_folder_path(segment_length = None, overlap = None):
        segment_length, overlap = GlobalVars.get_segment_lenght_and_overlap(segment_length, overlap)
            
        return GlobalVars.dataset_path + f"extended_segments_{segment_length}_sec_overlap_{overlap}" + os.sep
    
    def extended_dataset_file_path(segment_length = None, overlap = None):
        return GlobalVars.extended_dataset_folder_path(segment_length, overlap) + 'dataset_subsegments.csv'
    
    def train_file(suffix, segment_length = None, overlap = None):
        segment_length, overlap = GlobalVars.get_segment_lenght_and_overlap(segment_length, overlap)
        return GlobalVars.dataset_path + f"extended_segments_{segment_length}_sec_overlap_{overlap}" + os.sep + f"train_dataset_{suffix}.csv"

    def test_file(suffix, segment_length = None, overlap = None):
        segment_length, overlap = GlobalVars.get_segment_lenght_and_overlap(segment_length, overlap)
        return GlobalVars.dataset_path + f"extended_segments_{segment_length}_sec_overlap_{overlap}" + os.sep + f"test_dataset_{suffix}.csv"

    def validation_file(suffix, segment_length = None, overlap = None):
        segment_length, overlap = GlobalVars.get_segment_lenght_and_overlap(segment_length, overlap)
        return GlobalVars.dataset_path + f"extended_segments_{segment_length}_sec_overlap_{overlap}" + os.sep + f"validation_dataset_{suffix}.csv"
           
    def get_collection_name_for_experiment(cfg):
        segment_length, overlap = GlobalVars.get_segment_lenght_and_overlap()        
        features = '_'.join(cfg._FEATURES)
        collection_name = GlobalVars.get_cache_name_for_experiment(cfg) + f"_l{segment_length}_o{overlap}"
        return collection_name
    
    def get_cache_name_for_experiment(cfg, cache_type=""):
        return f"_{cache_type}_vectors_{cfg._ALL_CFG_FEATURES}_{cfg._VECTOR_OPERATION}"
         
        
    def get_cache_path_for_experiment(cfg, cache_type=""):
        folder_path = GlobalVars.extended_dataset_folder_path(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
        return folder_path + GlobalVars.get_cache_name_for_experiment(cfg, cache_type) + ".cache"
    
    def get_scaler_path_for_experiment(cfg, cache_type=""):
        folder_path = GlobalVars.extended_dataset_folder_path(cfg._SEGMENT_LENGHT, cfg._SEGMENT_OVERLAP)
        return folder_path + GlobalVars.get_cache_name_for_experiment(cfg, cache_type) + ".scaler"

    def get_all_overlaps_available(segment_lenght):
        path = GlobalVars.extended_dataset_folder_path(segment_length=segment_lenght, overlap = "*")
        tmp_str = path[:-2]
        
        overlaps = []
        folders = glob.glob(path)
        for f in folders:
            if os.path.isdir(f):
                last_part = f.replace(tmp_str, "") #f.split('/')[-2]  # 'extended_segments_10_sec_overlap_5'
                last_part = last_part[0:-1]
                if (last_part.isdigit()):
                    overlaps.append(int(last_part))
                
        return overlaps
RANDOM_SEED = 42
def init_random_seed(random_seed):
    global RANDOM_SEED
    if (random_seed > 0):
        RANDOM_SEED = random_seed
        
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    
    # enable_full_determinism(seed)
    #set_seed(seed)
    
def get_queen_status_description_map():
    return {
        0:'queen present or original queen',
        1:'queen not present',
        2 :'queen present and rejected', 
        3 :'queen present and newly accepted'
    }
    
    
def save_processed_data(df):
    df.to_csv(GlobalVars.csv_data_processed_path, index=False)
      
def sync_metadata_with_audio(show_df_stats:bool = True, save_processed = False,):
    """
    Loads the original file and operate some processing actions on it and returns the new dataframe:
    - eliminate not necessary columns ('lat','long', 'rain', 'gust speed', 'weatherID', 'time' )
    - created column file_name_prefix that represent the file name prefix for all audio files related with this row
    - create colum that specificy if the file is present or not on disk
    - create columns with label description (queen present or original queen / queen not present / queen present and rejected / queen present and newly accepted)
    - creates extra time-related columns ()
    Args:
        show_df_stats (bool, optional): Defaults to True, if TRUE, display dataframe head. 
        save_processed (bool, optional): Defaults to False, if TRUE save processed datafram.

    Returns:
        dataframe: processed dataframe
    """
    df=pd.read_csv(GlobalVars.csv_data_original, parse_dates=['date'], )
               
    if (show_df_stats):
        print(df.head())        
        print(df.nunique())
    
    #clean non important cols
    df=df.drop(['lat','long', 'rain', 'gust speed', 'weatherID', 'time' ],axis=1)
    df = df.sort_values(by='file name', ascending=True)
    
    #Each audio file has been segmented into 60-second files, so the file name will not match exactly. 
    #To match the audio clip to the correct row, isolate the prefix of the file name and filter through the wav files accordingly.
    df['file_name_prefix'] = df['file name'].str[:22]

    
    #count for each prefix how many segments were found
    df['segments_found'] = df['file_name_prefix'].apply(
        lambda prefix: sum(1 for file in os.listdir(GlobalVars.soundfiles_path) if file.startswith(prefix))
    )

    df['queen status desc']=df["queen status"].replace(get_queen_status_description_map())

    df = df[df['segments_found'] > 0]
    
    if (show_df_stats):
        total_files = sum(df['segments_found'])
        print(f"Segments calculated: {total_files}")

        df.head()

    if (save_processed == True):
        save_processed_data(df)
        
    df['record_timestamp'] = pd.to_datetime(df['date'])
    df['record_date'] = df['record_timestamp'].dt.date
    df['record_time'] = df['record_timestamp'].dt.time
    df['record_weekday'] = df['record_timestamp'].dt.weekday
    df['hour_of_day'] = df['record_timestamp'].dt.hour
    
    synch_df = sync_files_with_data(df)
    return synch_df


def sync_files_with_data(df_processed):
    """
    Synchronizes a preprocessed DataFrame with the available sound files in the dataset (folder).
    
    This function ensures a one-to-one correspondence between the entries in the DataFrame 
    and the sound files present in the dataset. The resulting synchronized DataFrame is saved to disk.

    Args:
        df_processed (pd.DataFrame): A preprocessed DataFrame containing metadata about the dataset.
                                     It must include a 'file_name_prefix' column to identify files.

    Returns:
        pd.DataFrame: A new DataFrame that is aligned with the actual sound files in the dataset.
    """

    new_rows = []

    #iterate each row from initial dataframe
    for _, row in df_processed.iterrows():
        file_name_prefix = row["file_name_prefix"]
        #found files in dataset that have the file-name starting with 'fine_name_prefix'
        files = [f for f in os.listdir(GlobalVars.soundfiles_path) if f.startswith(file_name_prefix)]
        #creates new rows for each segment in the list
        for file in files:
            new_row = row.to_dict()
            new_row["segment_file"] = file
            new_rows.append(new_row)

    #creates new dataframe based on data from list and save it to disk
    new_df = pd.DataFrame(new_rows)
    new_df = new_df.sort_values(by='segment_file', ascending=True)
    
    new_df = splitdata_unbalanced(new_df)
    new_df = splitdata_balanced(new_df)
    new_df.to_csv(GlobalVars.csv_data_sync_path, index=False)
    return new_df

def splitdata_unbalanced(df):
    df['train1'] = 0

    #process each class separatelly and separate train & test in equal proportion
    for status in df['queen status'].unique():
        subset = df[df['queen status'] == status]
        unique_file_name_prefix = subset['file_name_prefix'].unique()
        train_segments, test_segments = train_test_split(unique_file_name_prefix, test_size=TESTING_PERCENT, random_state=RANDOM_SEED)
        
        # set train1 = 1 for rows from trainset
        df.loc[df['file_name_prefix'].isin(train_segments) & (df['queen status'] == status), 'train1'] = 1
    return df

def splitdata_balanced(df):
    df['train2'] = 0

    min_segments = df.groupby('queen status')['file_name_prefix'].nunique().min()    
    train_set_size = int((1-TESTING_PERCENT) * min_segments)  # 80% from founded minimum
    print(f"min segments={min_segments}#{train_set_size}")

    for status in df['queen status'].unique():
        subset = df[df['queen status'] == status]
        unique_file_name_prefix = subset['file_name_prefix'].unique()
        
        train_segments = pd.Series(unique_file_name_prefix).sample(n=train_set_size, random_state=RANDOM_SEED).tolist()
        df.loc[df['file_name_prefix'].isin(train_segments) & (df['queen status'] == status), 'train2'] = 1
    return df
        
def mark_train_records_in_dataframe(new_df, save_path = False):
    df_original=pd.read_csv(GlobalVars.csv_data_sync_path )
    new_df = new_df.merge(df_original[['segment_file', 'train1', 'train2']], on='segment_file', how='left', suffixes=(None, '_new'))

    for col in ['train1', 'train2']:
        if f"{col}_new" in new_df.columns:
            new_df[col] = new_df[f"{col}_new"]
            new_df.drop(columns=[f"{col}_new"], inplace=True)
    
    if (save_path):
        if (os.path.exists(save_path)):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_path = save_path.replace(".csv", f"_backup_{backup_timestamp}.csv")

            shutil.copy(save_path, backup_file_path)
        new_df.to_csv(save_path, index=False)
    
    #print(new_df.head())
    return new_df
    
def save_extended_data(df_extended, segment_length, overlap):
    extended_df_folder_path = GlobalVars.extended_dataset_folder_path(segment_length = segment_length, overlap = overlap)    
    os.makedirs(extended_df_folder_path, exist_ok=True)
    extend_df_file_path = GlobalVars.extended_dataset_file_path(segment_length = segment_length, overlap = overlap)        
    df_extended.to_csv(extend_df_file_path, index=False)
    print(f"Extended file saved into: {extend_df_file_path}")
    
def load_original_data():
    df=pd.read_csv(GlobalVars.csv_data_processed_path, parse_dates=['date'], )
    
    df['record_timestamp'] = pd.to_datetime(df['date'])
    df['record_date'] = df['record_timestamp'].dt.date
    df['record_time'] = df['record_timestamp'].dt.time
    df['record_weekday'] = df['record_timestamp'].dt.weekday
    df['hour_of_day'] = df['record_timestamp'].dt.hour

    return df

def load_sync_files_data():
    df=pd.read_csv(GlobalVars.csv_data_sync_path, parse_dates=['date'], )
    
    return df

def generate_segmentation_metadata(df, segment_length, overlap, recreate_if_exists):
    """
    Extends the dataset by taking in consideration that each sound file will be splitted into smaller segments.

    This function processes the provided DataFrame by segmenting audio files into smaller parts 
    based on the specified segment length and overlap. Additionally, it will reduce the number of columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing metadata about the sound files.
        segment_length (int, optional): Length of each audio segment in seconds. Defaults to 10.
        overlap (int, optional): Overlapping duration between segments in seconds. Defaults to 5.

    Returns:
        pd.DataFrame: A new DataFrame with segmented audio data, saved to the appropriate location.
    """
    extend_df_file_path = GlobalVars.extended_dataset_file_path(segment_length = segment_length, overlap = overlap)
    if (os.path.exists(extend_df_file_path) and (recreate_if_exists == False)):
        df = pd.read_csv(extend_df_file_path)
        return df

    sound_file_length = 60
    segments = []
    for start in range(0, sound_file_length - segment_length + 1, segment_length - overlap):
        end = start + segment_length
        segments.append((start, end))

    new_rows = []

    for _, row in df.iterrows():
        base_file_name = row['segment_file'].replace(".wav", "")
        segment_no = row["segment_file"].replace(row['file_name_prefix']+ '__segment', '').replace(".wav", "")
        for i, (start, end) in enumerate(segments):
            new_row = {}
            new_row["file_name_prefix"] = row['file_name_prefix']
            new_row["segment_file"] = row['segment_file']
            new_row["queen presence"] = row['queen presence']
            new_row["queen acceptance"] = row['queen acceptance']
            new_row["queen status"] = row['queen status']
            new_row["segment_no"] = segment_no
            new_row["sub_segment_name"] =f"{base_file_name}_{start}_{end}"  
            new_row["sub_segment_no"] = i
            new_row["start_time"] = start
            new_row["end_time"] = end
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df = mark_train_records_in_dataframe(new_df)
    save_extended_data(new_df, segment_length=segment_length, overlap=overlap)    
    return new_df
    
def build_training_data_from_extended_data(balanced_type):
    extend_df_file_path = GlobalVars.extended_dataset_file_path()        
    training_file_path = GlobalVars.extended_dataset_folder_path() + f"training_{balanced_type}.csv"
    if (os.path.exists(training_file_path)):
        df_training=pd.read_csv(training_file_path)
    else:
        df_extended=pd.read_csv(extend_df_file_path )
        df_training = df_extended[(df_extended[f'train{balanced_type}'] ==1) & (df_extended['segment_no'] != 0) & (df_extended['segment_no'] != 5)]
        df_training.to_csv(training_file_path, index=False)
    return df_training, training_file_path

def build_testing_data_from_extended_data(balanced_type, include_first_segment = False, include_last_segment = False, include_regular_segments = True):
    extend_df_file_path = GlobalVars.extended_dataset_file_path()        
    testing_file_path = GlobalVars.extended_dataset_folder_path() + f"testing_{balanced_type}_{1 if include_first_segment else 0}_{1 if include_last_segment else 0}_{1 if include_regular_segments else 0}.csv"
    
    if (os.path.exists(testing_file_path)):
        df_testing=pd.read_csv(testing_file_path)
    else:
        df_extended=pd.read_csv(extend_df_file_path )
        df_testing = df_extended[df_extended[f'train{balanced_type}'] ==0]
        
        conditions = []

        if include_first_segment:
            conditions.append(df_testing['segment_no'] == 0)
        
        if include_last_segment:
            conditions.append(df_testing['segment_no'] == 5)
        
        if include_regular_segments:
            conditions.append(df_testing['segment_no'].isin([1, 2, 3, 4]))

        if conditions:
            df_testing = df_testing[pd.concat(conditions, axis=1).any(axis=1)] 
                    
        df_testing.to_csv(testing_file_path, index=False)
        
    return df_testing, testing_file_path
    
def generate_statistics_with_histogram(df, column_name, bins=1, title = None):
    
    stats_data = [
        {"Statistic": "Valid", "Value": df[column_name].count()},
        {"Statistic": "Missing", "Value": df[column_name].isnull().sum()},
        {"Statistic": "Mean", "Value":  "{:.2f}".format(df[column_name].mean())},
        {"Statistic": "Std. Deviation", "Value": "{:.2f}".format(df[column_name].std())},
        {"Statistic": "Min", "Value": df[column_name].min()},
        {"Statistic": "25%", "Value": df[column_name].quantile(0.25)},
        {"Statistic": "50% (Median)", "Value": df[column_name].median()},
        {"Statistic": "75%", "Value": df[column_name].quantile(0.75)},
        {"Statistic": "Max", "Value": df[column_name].max()},
    ]

    # Convertim lista într-un DataFrame pentru afișarea tabelului
    stats_df = pd.DataFrame(stats_data)
    
    # Creăm figura cu subgrafice
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Afișăm histograma
    sns.histplot(df[column_name], bins=bins, kde=False, ax=axes[0], edgecolor='black', color='blue',  discrete=True, alpha=0.7, )
    axes[0].set_title(f'Distribution of {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', linestyle='-', alpha=0.1)
   
    
    # Afișăm tabelul de statistici
    axes[1].axis('off')  # Ascundem axele pentru tabel
    # Adăugăm tabelul, aliniind valorile la dreapta
    table = axes[1].table(cellText=stats_df.values,
                 colLabels=stats_df.columns,
                 cellLoc='left',
                 loc='right')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Ajustăm dimensiunea tabelului


    # Aliniere la dreapta pentru valorile din coloana "Value"
    for i in range(len(stats_data)):
        table[i + 1, 1].set_text_props(ha="right")  # Aliniere la dreapta a valorilor
    
    if (title): 
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()    
    plt.show()
    
    
def plot_comparative_inside_outside(df, field, title):
    fig = plt.figure(figsize=(15, 12))

    # First subplot
    axes = fig.add_subplot(2, 1, 1)

    if (field == "temp"):
        inside_field_name = 'hive temp'
        outside_field_name = 'weather temp'
        ylabel = 'Temperatura (°C)'
    elif (field == "humidity"):
        inside_field_name = 'hive humidity'
        outside_field_name = 'weather humidity'
        ylabel = 'Umiditate %'
    elif (field == "pressure"):
        inside_field_name = 'hive pressure'
        outside_field_name = 'weather pressure'
        ylabel = 'Presiune'
        
    df_data = df[['date', inside_field_name, outside_field_name]].copy()
    df_data = df_data.dropna(how='all')
    if (field == "pressure"):
        #df_data = df_data[df_data[inside_field_name] != 0] and df_data[df_data[outside_field_name] != 0]
        #df_data = df_data[~(df_data[[inside_field_name, outside_field_name]] == 0).all(axis=1)]
        df_data[inside_field_name] = df_data[inside_field_name]-1000
        df_data[outside_field_name] = df_data[outside_field_name]-1000
        df_data = df_data[(df_data[inside_field_name] >= 0) & (df_data[outside_field_name] >= 0)]
    
    inside_data = df_data[inside_field_name]
    outside_data = df_data[outside_field_name]
    # Grafica pentru Temperatură
    axes.plot(df_data.index, inside_data, label='Temperatura Interior', color='orange')
    axes.plot(df_data.index, outside_data, label='Temperatura Exterior', color='blue')
    axes.set_xlabel('')
    axes.set_ylabel(ylabel)
    
    num_labels = 70  # Numărul de etichete pe care vrem să le afișăm pe axa X
    tick_indices = df_data.index[::len(df)//num_labels]  # Interval de selecție
    tick_labels = df_data['date'].dt.strftime('%d %b %H:%M')[::len(df)//num_labels]
    
    axes.set_xticks(tick_indices)
    axes.set_xticklabels(tick_labels, rotation=90, fontsize=8, fontweight='light')  # Etichete pe verticală
    axes.legend()

    
    # Titlu general pentru figura întreagă
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustăm layout-ul pentru a nu suprapune titlul

    plt.show()
    
def correlation_matrix(df):
    df_corelation = df[['hive number', 'hive temp', 'weather temp', 'hive humidity', 'weather humidity', 'hive pressure', 'weather pressure', 'wind speed', 
                        'record_weekday', 'cloud coverage', 'hour_of_day', 'queen acceptance', 'queen status']].copy()
       
    correlation_matrix = df_corelation.corr()    

    
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0, vmax=1.0, square=True, ax=ax, annot=True)

def multiple_corelation(df):
    data = df[['hour_of_day', 'hive temp', 'weather temp', 'hive humidity', 'weather humidity', 'hive pressure', 'weather pressure', 'wind speed', 'queen acceptance']].copy() 
  
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.3})
    plt.show()
    
def load_one_feature(feature_name, feature_file_name, features_path = None):
    if (features_path is None):
        features_path = GlobalVars.features_path(segment_length=GlobalVars.segment_length, overlap=GlobalVars.overlap, feature_name = feature_name) 
    feature_path = features_path + feature_file_name + ".npz"
    feature_data = np.load(feature_path)
    feature_data = feature_data['data']  
    return feature_data    

def load_one_feature_from_db(feature_name, sub_segment_name, db_cursor = None):
    if (db_cursor is None):
        db_path = GlobalVars.feature_db_path(segment_length=GlobalVars.segment_length, overlap=GlobalVars.overlap, feature_name = feature_name) 
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        db_cursor = conn.cursor()
    
    db_cursor.execute("SELECT vector FROM features WHERE sub_segment_name = ?", (sub_segment_name,))
    row = db_cursor.fetchone()

    if (db_cursor is None):
        conn.close()
    
    if row:
        buffer = io.BytesIO(row[0])
        vector = np.load(buffer)["data"]
        return vector
    return None 

def normalize_mfcc_data(mfcc_data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # normalizare între 0 și 1
    return scaler.fit_transform(mfcc_data.reshape(-1, mfcc_data.shape[-1])).reshape(mfcc_data.shape)

def normalize_mfcc_data_standard(mfcc_data):
    scaler = StandardScaler()
    return scaler.fit_transform(mfcc_data)

# def normalize_mfcc_data(mfcc_data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
    
#     mfcc_data_reshaped = mfcc_data.T
#     normalized_data = scaler.fit_transform(mfcc_data_reshaped)
    
#     return normalized_data.T 

def load_mel_data(df):
    mel_spectrums = []
    for file_name_prefix in tqdm( df['sub_segment_name'], total=df.shape[0]):
        melspec = load_one_mel(file_name_prefix)       
        mel_spectrums.append(melspec)
    return np.array(mel_spectrums)

def load_mfcc_data(df):
    mfccs = []
    for file_name_prefix in tqdm( df['sub_segment_name'], total=df.shape[0]):
        mfcc = load_one_mfcc(file_name_prefix)       
        mfccs.append(mfcc)
    return np.array(mfccs)

# def normalize_vector(vector):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler.fit(np.array([[-335], [230]]))  # Fixăm domeniul [-10, 10]
#     final_vector = scaler.fit_transform(vector.reshape(-1, 1)).flatten()
#     return final_vector

def reduce_vector(vector, n_components):
    flattened_vector = vector.T  

    pca = PCA(n_components=n_components)
    reduced_vector = pca.fit_transform(flattened_vector)  # Obținem (431, 64)

    final_vector = reduced_vector.flatten()
        
    return final_vector

def mean_vector(vector):
    final_vector = np.mean(vector, axis=1)
    return final_vector

def mean_split_vector(mfcc, n_splits=4, mean_iqr = None):
    n_mfcc, n_frames = mfcc.shape
    split_size = n_frames // n_splits
    vectors = []

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else n_frames
        segment = mfcc[:, start:end]
        if (mean_iqr is None):
            mean = np.mean(segment, axis=1)
        else:
            mean = mean_with_iqr_vector(segment, percent=mean_iqr)
        vectors.append(mean)

    result = np.concatenate(vectors)
    return result

def mean_with_iqr_vector(vector, percent = 25):
    Q1 = np.percentile(vector, percent, axis=1, keepdims=True)  # percentil 25%
    Q3 = np.percentile(vector, 100-percent, axis=1, keepdims=True)  # percentil 75%
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_vector = np.where((vector >= lower_bound) & (vector <= upper_bound), vector, np.nan)  # replace_outliers with NaN
    final_vector = np.nanmean(filtered_vector, axis=1)  # Calculăm media ignorând NaN
    return final_vector

def prepare_vector_for_db(vector, vector_operation, reduce_to_dim = False ):
    if (vector_operation == 'mean'):
        final_vector = mean_vector(vector)
    elif (vector_operation == 'pca'):
        final_vector = reduce_vector(vector, reduce_to_dim)
    elif (vector_operation == 'mean_iqr') or  (vector_operation == 'mean_iqr25'):
        final_vector = mean_with_iqr_vector(vector = vector, percent=25)
    elif (vector_operation == 'mean_iqr10'):
        final_vector = mean_with_iqr_vector(vector = vector, percent=10)
    elif (vector_operation == 'mean_iqr15'):
        final_vector = mean_with_iqr_vector(vector = vector, percent=15)
    elif (vector_operation == 'mean_split_2'):
        final_vector = mean_split_vector(vector, n_splits=2)
    elif (vector_operation == 'mean_split_3'):
        final_vector = mean_split_vector(vector, n_splits=3)
    elif (vector_operation == 'mean_split_4'):
        final_vector = mean_split_vector(vector, n_splits=4)
    elif (vector_operation == 'mean_iqr15_split_2'):
        final_vector = mean_split_vector(vector, n_splits=2, mean_iqr=15)
    elif (vector_operation == 'mean_iqr25_split_2' or vector_operation == 'mean_iqr_split_2'):
        final_vector = mean_split_vector(vector, n_splits=2, mean_iqr=25)
    elif (vector_operation == 'mean_iqr15_split_3'):
        final_vector = mean_split_vector(vector, n_splits=3, mean_iqr=15)
    elif (vector_operation == 'mean_iqr25_split_3'):
        final_vector = mean_split_vector(vector, n_splits=3, mean_iqr=25)
    else:
        print(f"Invalid operation: {vector_operation}")
        raise Exception(f"Invalid operation: {vector_operation}")
    return final_vector


def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acuratete Antrenament')
    plt.plot(history.history['val_accuracy'], label='Acuratete Validare')
    plt.title('Acuratete Antrenament și Validare')
    plt.xlabel('Epoci')
    plt.ylabel('Acuratete')
    plt.legend()

    # # Pierdere
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Pierdere Antrenament')
    # plt.plot(history.history['val_loss'], label='Pierdere Validare')
    # plt.title('Pierdere Antrenament și Validare')
    # plt.xlabel('Epoci')
    # plt.ylabel('Pierdere')
    # plt.legend()

    plt.show()
    
def params_to_md5(params):
    if (type(params) == 'dict' or isinstance(params, dict)):
        params_str = json.dumps(params) 
    else:
        params_str = params
    
    md5_hash = hashlib.md5()
    md5_hash.update(params_str.encode('utf-8'))
    return md5_hash.hexdigest(), params_str

def get_configurations_from_results(results_file_path, top_n):
    df=pd.read_csv(results_file_path)
    df = df.dropna(subset=["accuracy"])
    df_sorted = df.sort_values(by="accuracy", ascending=False)
    return df_sorted.head(top_n)

def display_one_param_pie(field_name, df_all, df_top1, df_top2, n_top1, n_top2, title):
    # Contorizează frecvențele funcțiilor de activare pentru ambele seturi
    groups_all = df_all[field_name].value_counts()
    groups_top2 = df_top2[field_name].value_counts()
    groups_top1 = df_top1[field_name].value_counts()

    
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    axes[0].pie(groups_all, labels=groups_all.index, autopct='%1.1f%%', startangle=140)
    axes[0].set_title(f"{title} \n Set complet")


    axes[1].pie(groups_top2, labels=groups_top2.index, autopct='%1.1f%%', startangle=140)
    axes[1].set_title(f"{title} \n Top {n_top2} Accuracy")


    axes[2].pie(groups_top1, labels=groups_top1.index, autopct='%1.1f%%', startangle=140)
    axes[2].set_title(f"{title} \n Top {n_top1} Accuracy")

    # Afișează graficele
    plt.tight_layout()
    plt.show()
    
    max_label = groups_top1.idxmax()  
    max_value = groups_top1.max() 
    print(f"Top {n_top2}:   {groups_top2.idxmax()}: {groups_top2.max()}%")
    print(f"Top {n_top1}:   {groups_top1.idxmax()}: {groups_top1.max()}%")
    return max_label

def plot_2levels_pie(df, group1_criteria, group2_criteria, ax, title):
    grouped = df.groupby([group1_criteria, group2_criteria]).size().reset_index(name="count")
    
    
    outer_labels = grouped[group1_criteria].unique()
    outer_sizes = grouped.groupby(group1_criteria)["count"].sum().values
    inner_labels = [f"{row[group1_criteria]} - {row[group2_criteria]}" for _, row in grouped.iterrows()]
    inner_sizes = grouped["count"].values
    
   
    size = 0.4
    cmap = plt.colormaps["tab20c"]
    
    outer_colors = [cmap(i*4  ) for i in range(len(outer_labels))]
    inner_colors = [outer_colors[list(outer_labels).index(act)] for act in grouped[group1_criteria]]
    inner_colors = cmap([1, 2, 5, 6, 9, 10])
    
    
    ax.pie(outer_sizes, labels=outer_labels, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'), autopct='%1.1f%%',  pctdistance=0.85, startangle=140)   
    ax.pie(inner_sizes, labels=inner_labels, radius=1-size, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w'), autopct='%1.1f%%', startangle=140)
    
    ax.set_title(title)
    
def display_2params_pie(param1, param2, df_all, df_top1, df_top2, n_top1, n_top2, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    plot_2levels_pie(df_all, param1, param2, axes[0], f"{title} \n Set complet")
    plot_2levels_pie(df_top2, param1, param2, axes[1], f"{title} \n Set complet - Top {n_top2} mape")
    plot_2levels_pie(df_top1, param1, param2, axes[2], f"{title} \n Set complet - Top {n_top1} mape")

    plt.tight_layout()
    plt.show()
    
def load_resuls_file(results_file_path, sort_by = None, sort_dir = None):
    df=pd.read_csv(results_file_path)
    df = df.dropna(subset=["accuracy"])
    if (sort_by is None):
        sort_by = ["accuracy", "feature", "vector_operation"]
        sort_dir = [False, True, True]
    df_sorted = df.sort_values(by=sort_by, ascending=sort_dir)
    return df_sorted

def print_results_stat(df_sorted, n_top1, n_top2,):
    print_top = df_sorted.head(n_top1).copy()
    cols_to_drop = ['skipped', 'training_set_size', 'testing_set_size', 'balanced_type', 'row_key']
    for col in cols_to_drop:
        try:
            if col in print_top.columns:
                print_top = print_top.drop(col, axis=1)
        except Exception as e:
            print(f"Eroare la ștergerea coloanei '{col}': {e}")
        
    print(print_top.to_string(index=False)) 
    print("\n\n")
    
    df_top1 = df_sorted.head(n_top1)
    df_top2 = df_sorted.head(n_top2)
    #print(df_top100.to_string(index=False)) 
    print(f"Experimente:\t\t {len(df_sorted)}")
    print(f"Max:\t\t {max(df_sorted['accuracy'])}")
    print(f"Avg {n_top1}:\t {df_top1['accuracy'].mean()}")
    print(f"Min {n_top1}:\t {min(df_top1['accuracy'])}")
    print(f"Avg {n_top2}:\t {df_top2['accuracy'].mean()}")
    print(f"Min {n_top2}:\t {min(df_top2['accuracy'])}")
    print(f"Avg:\t\t {df_sorted['accuracy'].mean()}")
    print(f"Min:\t\t {min(df_sorted['accuracy'])}")
    return df_top1, df_top2

def display_confusion_matrix(cm, labels = None, title = ''):
    if (labels is None):
        labels_desc = get_queen_status_description_map()
        labels = [labels_desc[0], labels_desc[1], labels_desc[2], labels_desc[3]]
    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=90)
    plt.title(title)
    plt.show()

def display_classification_report(classification_report, labels_desc = None):
    if (labels_desc is None):
        labels_desc = get_queen_status_description_map()
    
    classification_report_renamed = {
        labels_desc[int(k)] if k.isdigit() else k: v
        for k, v in classification_report.items()
    }

    df_report = pd.DataFrame(classification_report_renamed).T
    df_report = df_report.round(4) #round to 4 decimals
    print(df_report)
    
def mem_MB():
    return process.memory_info().rss / 1024 / 1024  # în MB

def mem_KB():
    return process.memory_info().rss / 1024   # în KB

def extract_experiment_results(new_row, tool_results, process_key, error):
    if (tool_results is not None):
        classification_report = tool_results['classification_report']
        new_row["skipped"] = False
        new_row['neighbors'] = tool_results["neighbors"]
        new_row['vote_type'] = tool_results["vote_type"]
        new_row['accuracy'] = tool_results["accuracy_score"]
        for idx in range(4):
            new_row[f'precision_{idx}'] = classification_report[f'{idx}']['precision']
        new_row['training_set_size'] = tool_results["training_set_size"]
        new_row['testing_set_size'] = tool_results["testing_set_size"]
        new_row['train_elapsed_time'] = tool_results["train_elapsed_time"]
        new_row['predict_elapsed_time'] = tool_results["predict_elapsed_time"]
        new_row['train_used_memory'] = tool_results["train_used_memory"]
        new_row['predict_used_memory'] = tool_results["predict_used_memory"]
        new_row['row_key'] = process_key
    else:
        new_row["skipped"] = True
        print("Error on tool execution:", error)
    return new_row
    
def measure_perf(func, *args, **kwargs):
    peak_mem = 0
    running = True

    def memory_tracker():
        nonlocal peak_mem
        while running:
            rss = process.memory_info().rss
            peak_mem = max(peak_mem, rss)
            time.sleep(0.01)  # check memory every 10ms

    # start monitoring
    monitor_thread = threading.Thread(target=memory_tracker)
    monitor_thread.start()

    # start time measure and call function
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time

    # oprește monitorizarea
    running = False
    monitor_thread.join()

    peak_mem_MB = peak_mem / 1024 / 1024

    return {
        'result': result,
        'elapsed_time': elapsed_time,
        'peak_memory_MB': peak_mem_MB
    }
    
def concat_csvs(csvs, dest_cst):
    if not csvs:
        raise ValueError("LEmpty list")

    all_dfs = []
    for idx, csv_file in enumerate(csvs):
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error on reading file: {csv_file}: {e}")

    if not all_dfs:
        raise ValueError("Files not loaded!")

    df_total = pd.concat(all_dfs, ignore_index=True)
    df_total.to_csv(dest_cst, index=False)

    return df_total


def majority_vote(labels):
    return np.bincount(labels).argmax()


def majority_vote_with_score(row):
    counts = np.bincount(row)
    label = counts.argmax()
    score = counts[label] / len(row)
    return label, score

def combine_multiple_parameters_v3(combinations_file_path, index_params_list):
    segment_prop = [(10,'all')]
    features = ['pe-mfcc_40']
    operations = ['mean', 'mean_iqr25']
    metric_types = ['cosine', 'correlation']
    vote_types = ['uniform']
    neighbors_list = [15, 20]

    #all posible combinations
    param_combinations = list(product(
        segment_prop, features, operations, metric_types, vote_types, neighbors_list, index_params_list
    ))    

    all_configurations = [
        {
            "segment_lenght": segp[0],            
            "segment_overlap": segp[1],    
            "feature": feat,
            "vector_operation": operation,     
            "metric_type": metric_type,
            "vote_type": vote_type,
            "neighbors": neighbors,
            "index_params": index_params,                    
        }
        for segp, feat, operation, metric_type, vote_type, neighbors, index_params in param_combinations
    ]
    df_configuration = pd.DataFrame(all_configurations)
    df_configuration.to_csv(combinations_file_path, index=False) 