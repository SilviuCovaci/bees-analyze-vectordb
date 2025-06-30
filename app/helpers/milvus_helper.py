
from pymilvus import MilvusClient, DataType, Role, Collection, has_collection, CollectionSchema, FieldSchema, DataType
from pymilvus import connections, db
import shutil
import time

using_lite = 0
if (using_lite):
    from milvus import default_server
    
class MilvusHelper:
    
    server = None
    
    client = None
    connection = None
    
    debug_mode = False
    
    _TOKEN = None
    _DB_NAME = "default"
    _URI = "http://localhost:19530"
    _DATA_PATH="milvus_data"
    _HOST = '127.0.0.1'
    _PORT = 19530
    
    def __init__(self, debug_mode = False):
        self.debug_mode = debug_mode
        if using_lite:
            self._PORT = default_server.listen_port
            self._HOST = default_server.server_address
            MilvusHelper.server = default_server
            MilvusHelper.server.set_base_dir(self._DATA_PATH)
            self.debug(self._HOST)

            self.debug("Is server started:", MilvusHelper.server.running)
            if MilvusHelper.server.running is False:
                MilvusHelper.server.start()
            
            self.debug("Server is running ...", MilvusHelper.server.running)
        self.connect(debug_mode)

    def debug(self, *args, **kwargs):
        if (self.debug_mode):
            print(*args, **kwargs)
            
    def connect(self, debug = False):
        self.connection = connections.connect(host=self._HOST, port=self._PORT)
        self.debug("Self connection=", connections.list_connections())
        self.debug("Conexiunea cu Milvus este activă, for databases:", db.list_database())
            
    
    def get_client(self):
        if (not self.client is None):
            return self.client
        
        if (using_lite):
            self._URI = 'http://' + MilvusHelper.server.server_address + ":"  + str(MilvusHelper.server.listen_port)
        self.debug("Connect client to:", self._URI)
        self.client = MilvusClient(
            uri=self._URI,
            token=self._TOKEN,
            database=self._DB_NAME
        )        
        self.debug("Client active with collections",  self.client.list_collections())
        return self.client
    

    def create_vector_index(self, collection_obj, field_name, index_params):
        # index_params = {
        #     "index_type": "IVF_PQ",
        #     "metric_type": "L2",
        #     "params": {"nlist": n_list, "m": m, "nbit": nbit}
        # }
        milvus_index_type = index_params.get("params").get("index_type")
        milvus_metric_type = index_params.get("metric_type")
        milvus_index_params = index_params.get("params").get("params")
        self.debug(f"Create vector index with params: type:{milvus_index_type},metric_type:{milvus_metric_type};valoarea params:{milvus_index_params}")
        new_index_params = {
            "index_type": milvus_index_type,
            "metric_type": milvus_metric_type,
        }
        if (not milvus_index_params is None):
            new_index_params["params"] = milvus_index_params
        collection_obj.create_index(
            field_name=field_name,
            index_params=new_index_params,
            timeout=None,
            sync=True,
        )
    
    def wait_for_index_removal(self, collection_obj, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not collection_obj.indexes:
                return True
            self.debug("collection has index=", collection_obj.has_index())
            time.sleep(0.5)
        raise TimeoutError("Indexul nu s-a șters complet în timp util.")

    def recreate_vector_index(self, collection_obj, field_name, index_params, wait_till_ready=None, poll_interval=0.1):
        collection_obj.release()
        if (collection_obj.has_index()):
            self.debug("Drop Index")
            collection_obj.drop_index()    
            self.wait_for_index_removal(collection_obj, 10)
            self.debug("Done. Create the new one!")
            self.debug("Deleted:" + str(not collection_obj.has_index()))
        elapsed_time = 0
        start_time = time.perf_counter()

        try:
            #index_params["params"]["random_seed"] = 42
            
            milvus_index_type = index_params.get("params").get("index_type")
            milvus_metric_type = index_params.get("metric_type")
            milvus_index_params = index_params.get("params").get("params")
            self.debug(f"Create vector index with params: type:{milvus_index_type},metric_type:{milvus_metric_type};valoarea params:{milvus_index_params}")
            new_index_params = {
                "index_type": milvus_index_type,
                "metric_type": milvus_metric_type,
            }
            if (not milvus_index_params is None):
                new_index_params["params"] = milvus_index_params
            collection_obj.create_index(
                field_name=field_name,
                index_params=new_index_params,
                timeout=None,
                sync=True
            )
        except Exception as e:
            self.debug("Eroare la crearea indexului:", e)
            
        # print("Index creation triggered.")
        # if not wait_till_ready is None:
        #     print("Waiting for index to be ready...")
        #     while not collection_obj.has_index():
        #         elapsed_time = time.perf_counter() - start_time
        #         if elapsed_time > wait_till_ready:
        #             return False, elapsed_time
        #         time.sleep(poll_interval)
        #     print("Created!!!", elapsed_time, time.perf_counter() - start_time)
            
        return collection_obj.has_index(), time.perf_counter() - start_time
        
   
    def describe_index(self, collection_name):
        res = self.client.list_indexes(
            collection_name=collection_name
        )

        self.debug("Indexes:", res)

        res = self.client.describe_index(
            collection_name=collection_name,
            index_name="vector_data"
        )
        self.debug("Index Description:", res)
        
    def create_vector_collection(self, collection_name, field_name, vector_dimension, index_params):    
        primary_key = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        vector_field = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=vector_dimension)
        queen_status = FieldSchema(name="queen_status", dtype=DataType.INT8)
        queen_presence = FieldSchema(name="queen_presence", dtype=DataType.INT8)
        queen_acceptance = FieldSchema(name="queen_acceptance", dtype=DataType.INT8)
        segment_no = FieldSchema(name="segment_no", dtype=DataType.INT8)
        sub_segment_name = FieldSchema(name="sub_segment_name", dtype=DataType.VARCHAR, max_length=512)
        sub_segment_no = FieldSchema(name="sub_segment_no", dtype=DataType.INT8)        

        schema = CollectionSchema(
            fields = [primary_key, vector_field, queen_status, queen_presence, queen_acceptance, segment_no, sub_segment_name, sub_segment_no],
            enable_dynamic_field = True
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
        )
        
        self.create_vector_index(collection, field_name=field_name, index_params=index_params)
        return collection
    
    def insert_records(self, collection_name, records, do_flush = True):
        res = self.client.insert(
            collection_name=collection_name,
            data=records
        )
        
        if (do_flush):
            Collection(collection_name).flush()  
        return res
    
    def get_collection(self, collection_name):
        """
        Obtain a collection by name

        Args:
            collection_name (str): name of collection

        Returns:
            Collection: Milvus specific instance of collection
        """
        if has_collection(collection_name):
            return Collection(collection_name)
        else:
            return None
        
    def clear_collection(self, collection_name):
        """
        Clear the specified collection

        Args:
            collection_name (str): name of collection

        Returns:
            None
        """
        collection = Collection(collection_name)
        collection.delete(expr="id > 0")  
        collection.flush()  
        collection.release()
        collection.compact()
        
        num_records = collection.num_entities
        self.debug(f"Numărul de înregistrări din colecție: {num_records}")
        return collection
    
    def loaded_state(self, collection_name):
        res = self.client.get_load_state(collection_name=collection_name)
        
        self.debug("res=", res, type(res.get('state')))

        # Valorile posibile ale stării:
        # 1 - NotLoad
        # 2 - Loading
        # 3 - Loaded

        return res.get("state", -1)  # Obține starea sau -1 dacă lipsește
        
    def load_collection_if_need(self, collection_name):
        
        if self.loaded_state(collection_name) == 1: #Collection is not loaded
             self.client.load_collection(collection_name=collection_name)
                    

