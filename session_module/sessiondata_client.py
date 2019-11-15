import grpc
import session_module.sessiondata_pb2
import session_module.sessiondata_pb2_grpc


with grpc.insecure_channel('localhost:50051') as channel:
    stub = session_module.sessiondata_pb2_grpc.SessionDataServiceStub(channel)
    request = session_module.sessiondata_pb2.PingSessRequest()
    stub.PingSess(request)
    print("----------")
    request = session_module.sessiondata_pb2.SessionRequestInfo(last_hours=2)
    a = stub.GetSessionData(request)
    print(a)