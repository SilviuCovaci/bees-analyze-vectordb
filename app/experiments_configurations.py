flat_configs = ["{'params':'Flat'}", "{'params':'SQ8'}", "{'params':'SQfp16'}", "{'params':'PQ20'}", "{'params':'PQ8'}", "{'params':'PQ10'}"]

milvus_pq_10 = [
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 1000}, 'nprobe': 20},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 1000}, 'nprobe': 40},     
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 1000}, 'nprobe': 100},     
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 2000}, 'nprobe': 40},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 2000}, 'nprobe': 100},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 2000}, 'nprobe': 500},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 4000}, 'nprobe': 100},  
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 4000}, 'nprobe': 250},  
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 4000}, 'nprobe': 500},  
    {'index_type': 'IVF_PQ', 'params': {'m': 10, 'nlist': 4000}, 'nprobe': 1000},  
]
milvus_ivf_all = [
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 1000}, 'nprobe': 20},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 1000}, 'nprobe': 40},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 1000}, 'nprobe': 100},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 2000}, 'nprobe': 40},
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 2000}, 'nprobe': 100},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 2000}, 'nprobe': 500},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 4000}, 'nprobe': 100},
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 4000}, 'nprobe': 250},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 4000}, 'nprobe': 500},                    
    {'index_type': 'IVF_FLAT', 'params': {'nlist': 4000}, 'nprobe': 1000},                    
     
     
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 1000}, 'nprobe': 20},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 1000}, 'nprobe': 40},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 1000}, 'nprobe': 100},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 2000}, 'nprobe': 40},
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 2000}, 'nprobe': 100},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 2000}, 'nprobe': 500},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 4000}, 'nprobe': 100},
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 4000}, 'nprobe': 250},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 4000}, 'nprobe': 500},                    
    {'index_type': 'IVF_SQ8', 'params': {'nlist': 4000}, 'nprobe': 1000},
    
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 1000}, 'nprobe': 20},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 1000}, 'nprobe': 40},     
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 1000}, 'nprobe': 100},     
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 2000}, 'nprobe': 40},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 2000}, 'nprobe': 100},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 2000}, 'nprobe': 500},                    
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 4000}, 'nprobe': 100},  
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 4000}, 'nprobe': 250},  
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 4000}, 'nprobe': 500},  
    {'index_type': 'IVF_PQ', 'params': {'m': 8, 'nlist': 4000}, 'nprobe': 1000},  
]

ivf_all = [
        "{'params':'IVF1000,Flat', 'nprobe': 20}",
        "{'params':'IVF1000,Flat', 'nprobe': 40}",
        "{'params':'IVF1000,Flat', 'nprobe': 100}",
        "{'params':'IVF2000,Flat', 'nprobe': 40}",
        "{'params':'IVF2000,Flat', 'nprobe': 100}",
        "{'params':'IVF2000,Flat', 'nprobe': 500}",
        "{'params':'IVF4000,Flat', 'nprobe': 100}",
        "{'params':'IVF4000,Flat', 'nprobe': 250}",
        "{'params':'IVF4000,Flat', 'nprobe': 500}",
        "{'params':'IVF4000,Flat', 'nprobe': 1000}",

        "{'params':'IVF1000,SQ8', 'nprobe': 20}",
        "{'params':'IVF1000,SQ8', 'nprobe': 40}",
        "{'params':'IVF1000,SQ8', 'nprobe': 100}",
        "{'params':'IVF2000,SQ8', 'nprobe': 40}",
        "{'params':'IVF2000,SQ8', 'nprobe': 100}",
        "{'params':'IVF2000,SQ8', 'nprobe': 500}",
        "{'params':'IVF4000,SQ8', 'nprobe': 100}",
        "{'params':'IVF4000,SQ8', 'nprobe': 250}",
        "{'params':'IVF4000,SQ8', 'nprobe': 500}",
        "{'params':'IVF4000,SQ8', 'nprobe': 1000}",

                
        "{'params':'IVF1000,PQ10', 'nprobe': 20}",
        "{'params':'IVF1000,PQ10', 'nprobe': 40}",
        "{'params':'IVF1000,PQ10', 'nprobe': 100}",
        "{'params':'IVF2000,PQ10', 'nprobe': 40}",
        "{'params':'IVF2000,PQ10', 'nprobe': 100}",
        "{'params':'IVF2000,PQ10', 'nprobe': 500}",
        "{'params':'IVF4000,PQ10', 'nprobe': 100}",
        "{'params':'IVF4000,PQ10', 'nprobe': 250}",
        "{'params':'IVF4000,PQ10', 'nprobe': 500}",
        "{'params':'IVF4000,PQ10', 'nprobe': 1000}",        
]

hnsw_efsearch_50 = [
    "{'params':'HNSW50,Flat', 'efConstruction': 300,'efSearch': 50}",    
    "{'params':'HNSW50,Flat', 'efConstruction': 500,'efSearch': 50}",        
    "{'params':'HNSW100,Flat', 'efConstruction': 500,'efSearch': 50}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 700,'efSearch': 50}",    
    
    "{'params':'HNSW50,SQ8', 'efConstruction': 300,'efSearch': 50}",    
    "{'params':'HNSW50,SQ8', 'efConstruction': 500,'efSearch': 50}",  
    "{'params':'HNSW100,SQ8', 'efConstruction': 500,'efSearch': 50}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 700,'efSearch': 50}",  
    
    "{'params':'HNSW50,PQ10', 'efConstruction': 300,'efSearch': 50}",    
    "{'params':'HNSW50,PQ10', 'efConstruction': 500,'efSearch': 50}",  
    "{'params':'HNSW100,PQ10', 'efConstruction': 500,'efSearch': 50}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 700,'efSearch': 50}",  
]


hnsw_all = [
    
    "{'params':'HNSW50,Flat', 'efConstruction': 300,'efSearch': 90}",    
    "{'params':'HNSW50,Flat', 'efConstruction': 300,'efSearch': 180}",    
    "{'params':'HNSW50,Flat', 'efConstruction': 300,'efSearch': 300}",    
    "{'params':'HNSW50,Flat', 'efConstruction': 500,'efSearch': 90}",        
    "{'params':'HNSW50,Flat', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW50,Flat', 'efConstruction': 500,'efSearch': 300}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 500,'efSearch': 90}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 500,'efSearch': 300}",        
    "{'params':'HNSW100,Flat', 'efConstruction': 700,'efSearch': 90}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 700,'efSearch': 180}",    
    "{'params':'HNSW100,Flat', 'efConstruction': 700,'efSearch': 300}",  
    
    "{'params':'HNSW50,SQ8', 'efConstruction': 300,'efSearch': 90}",    
    "{'params':'HNSW50,SQ8', 'efConstruction': 300,'efSearch': 180}",    
    "{'params':'HNSW50,SQ8', 'efConstruction': 300,'efSearch': 300}",    
    "{'params':'HNSW50,SQ8', 'efConstruction': 500,'efSearch': 90}",        
    "{'params':'HNSW50,SQ8', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW50,SQ8', 'efConstruction': 500,'efSearch': 300}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 500,'efSearch': 90}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 500,'efSearch': 300}",        
    "{'params':'HNSW100,SQ8', 'efConstruction': 700,'efSearch': 90}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 700,'efSearch': 180}",    
    "{'params':'HNSW100,SQ8', 'efConstruction': 700,'efSearch': 300}",  

    "{'params':'HNSW50,PQ10', 'efConstruction': 300,'efSearch': 90}",    
    "{'params':'HNSW50,PQ10', 'efConstruction': 300,'efSearch': 180}",    
    "{'params':'HNSW50,PQ10', 'efConstruction': 300,'efSearch': 300}",    
    "{'params':'HNSW50,PQ10', 'efConstruction': 500,'efSearch': 90}",        
    "{'params':'HNSW50,PQ10', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW50,PQ10', 'efConstruction': 500,'efSearch': 300}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 500,'efSearch': 90}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 500,'efSearch': 180}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 500,'efSearch': 300}",        
    "{'params':'HNSW100,PQ10', 'efConstruction': 700,'efSearch': 90}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 700,'efSearch': 180}",    
    "{'params':'HNSW100,PQ10', 'efConstruction': 700,'efSearch': 300}",          
]
     





