
import sqlite3
import os, io
import numpy as np
import glob

from library import GlobalVars
import helpers.features_helper as features_helper

sound_files_path_custom = None

def array_to_blob(vector):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=vector)
    blob = sqlite3.Binary(buffer.getvalue())
    return blob

def extract_features_from_batch(df_partition, partition_info, global_config):
    
    #print(partition_info)
    #print(global_config)
    partition_id = partition_info["number"]
    features2extract = global_config["features"]
    segment_lenght = global_config["segment_lenght"]
    segment_overlap = global_config["overlap"]
    
    connections = {}
    cursors = {}
    data_to_insert = {}
    for feature in features2extract:
        db_path = GlobalVars.feature_db_path(segment_length=segment_lenght, overlap=segment_overlap, feature_name=feature, partition_id=partition_id)
        conn = sqlite3.connect(db_path, timeout=30)        
        
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                sub_segment_name TEXT PRIMARY KEY,
                vector BLOB
            )
        """)
        cursor.execute(f"DELETE FROM features;")
        connections[feature] = conn
        cursors[feature] = cursor
        data_to_insert[feature] = []
    
    for _, row in df_partition.iterrows():
        audio_path = (sound_files_path_custom if sound_files_path_custom else GlobalVars.soundfiles_path) + row['segment_file']    
        extracted_features  = features_helper.extract_features_from_audio(audio_path, start_time=row['start_time'], duration=segment_lenght, features=features2extract)
        for feature in features2extract:
            blob = array_to_blob(extracted_features[feature])
            data_to_insert[feature].append((row['sub_segment_name'], blob))
            
    for feature in features2extract:
        cursors[feature].executemany("INSERT OR REPLACE INTO features(sub_segment_name, vector) VALUES (?, ?)", data_to_insert[feature] )
        connections[feature].commit()
        connections[feature].close()
    
    return df_partition

def merge_partition_dbs(feature, segment_length, segment_overlap):
    final_db_path = GlobalVars.features_path(segment_length=segment_length, overlap=segment_overlap, feature_name=feature)  + ".db"        
    partition_db_files = glob.glob(
        GlobalVars.features_path(segment_length=segment_length, overlap=segment_overlap, feature_name=feature)  + f"_p*.db"        
    )
    
    # creates final db and table
    conn_final = sqlite3.connect(final_db_path)
    cursor_final = conn_final.cursor()
    cursor_final.execute("""
        CREATE TABLE IF NOT EXISTS features (
            sub_segment_name TEXT PRIMARY KEY,
            vector BLOB
        )
    """)
    conn_final.execute(f"DELETE FROM features;")
    conn_final.commit()

    conn_final.execute("BEGIN")
    try:
        idx = 0
        for part_db_path in partition_db_files:
            
            if not os.path.exists(part_db_path):
                print(f"Partition DB not found: {part_db_path}")
                continue

            conn_part = sqlite3.connect(part_db_path)
            cursor_part = conn_part.cursor()

            cursor_part.execute("SELECT sub_segment_name, vector FROM features")
            rows = cursor_part.fetchall()

            cursor_final.executemany("INSERT OR REPLACE INTO features VALUES (?, ?)", rows)
            
            conn_part.close()
            print(f"Done partition {idx}")
            idx  = idx + 1
        print("final commit")
        conn_final.commit()
    except Exception as e:
        conn_final.rollback()
        raise
    finally:
        conn_final.close()
    
def clear_partition_dbs(feature, segment_length, segment_overlap):
    partition_db_files = glob.glob(
        GlobalVars.features_path(segment_length=segment_length, overlap=segment_overlap, feature_name=feature)  + f"_p*.db"        
    )
    for part_db_path in partition_db_files:
        if os.path.exists(part_db_path):
            os.remove(part_db_path)
            
def merge_all_partitions(features2extract):
    for feature in features2extract:
        print(f"Process feature: {feature}")
        merge_partition_dbs(feature, GlobalVars.segment_length, GlobalVars.overlap)
        clear_partition_dbs(feature, GlobalVars.segment_length, GlobalVars.overlap)