import grpc
import logging
from concurrent import futures
import session_module.sessiondata_pb2
import session_module.sessiondata_pb2_grpc
from nar_module.nar.nar_trainer_cafebiz_full import get_session_features_config
from acr_module.acr.acr_module_service import load_json_config
from pick_singleton.pick_singleton import NAR_Pickle_Singleton,ACR_Pickle_Singleton
from nar_module.nar.datasets import prepare_dataset_iterator
from nar_module.nar.utils import resolve_files, chunks
import tensorflow as tf

class SessionDataServiceServicer(session_module.sessiondata_pb2_grpc.SessionDataServiceServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nar_label_encoders = NAR_Pickle_Singleton.getInstance()
        self.session_features_config = get_session_features_config(nar_label_encoders)
        self.batch_size = 64
        self.truncate_session_length = 20
    def process_data(self, training_hour):
        train_data = self.get_training_files(training_hour)
        # print(train_data)
        it = prepare_dataset_iterator(train_data, self.session_features_config, 
                                                                        batch_size=self.batch_size,
                                                                        truncate_session_length=self.truncate_session_length)
        count = 0
        with tf.Session() as sess:
            list_sessions = session_module.sessiondata_pb2.ListsSessions()
            # list_sessions = []
            while True:
                try:
                    data_it = sess.run(it)
                    sessions_batch = self.get_all_sessions_clicks(data_it[0]['item_clicked'],data_it[1]['label_next_item'])
                    # print("SESSION BATCH: "+ str(count))
                    # print(len(sessions_batch))
                    # print(sessions_batch)
                    self.convert_encoded_ids(sessions_batch)
                    # print("SESSION BATCH ENCODED: "+ str(count))
                    # print(len(sessions_batch))
                    # print(sessions_batch)
                    for session in sessions_batch:
                        # list_sessions.append(session)
                        session_msg = session_module.sessiondata_pb2.Session()
                        for item in session:
                            session_msg.session_item.append(str(item))
                        # print("SESSION MESSAGE")
                        # print(session_msg)
                        list_sessions.sessions.append(session_msg)
                    count +=1
                    # clf.train(data_it[0]['user_id'],data_it[0]['session_id'],
                    # data_it[0]['item_clicked'],data_it[1]['label_next_item'])
                except tf.errors.OutOfRangeError:
                    break   
        print("Total training sample: "+ str(count*self.batch_size))
        return list_sessions

    def get_training_files(self, training_hour):
        parameter  = load_json_config("./parameter.json")
        training_dir = parameter["DATA_DIR"]+ parameter["nar_preprocess_2"]["output_sessions_tfrecords_path"]
        train_files = resolve_files(training_dir)[-training_hour:]
        # print("TrainFile")
        # print(train_files)
        return list(chunks(train_files, training_hour))
    
    def convert_encoded_ids(self,clickeds_all_items):
            acr_pickle = ACR_Pickle_Singleton.getInstance()
            for i,clicked_items in enumerate(clickeds_all_items):
                for j,item in enumerate(clicked_items):
                    if item != 0:
                        clickeds_all_items[i][j] = acr_pickle.get_article_id(item)

    def get_all_sessions_clicks(self, sessions_items, sessions_next_items):
        sessions_all_items_but_last = list([list(filter(lambda x: x != 0, session)) for session in sessions_items])
        sessions_last_item_clicked = list([list(filter(lambda x: x != 0, session))[-1] for session in sessions_next_items])
        sessions_all_clicks = [previous_items + [last_item] \
                                for previous_items, last_item in zip(sessions_all_items_but_last, sessions_last_item_clicked)]
        return sessions_all_clicks
    def GetSessionData(self, request, context):
        # a = self.process_data(request.last_hours)
        # return session_module.sessiondata_pb2.ListsSessions(sessions = a)
        return self.process_data(request.last_hours)
    
    def PingSess(self, request, context):
        return session_module.sessiondata_pb2.PingSessResponse(data="PONG")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    session_module.sessiondata_pb2_grpc.add_SessionDataServiceServicer_to_server(
        SessionDataServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()