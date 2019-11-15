import redis
REDIS_SERVER_CONF = {
    'servers' : {
      'localhost': {
        'HOST' : 'localhost',
        'PORT' : 6379 ,
        'DATABASE':0
      },
      'cloud95': {
        'HOST' : '10.5.36.95',
        'PORT' : 6379 ,
        'DATABASE':0
      }
    
  }
}
class Singleton(type):
    """
    An metaclass for singleton purpose. Every singleton class should inherit from this class by 'metaclass=Singleton'.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
class RedisClient(object,metaclass=Singleton):

    def __init__(self,server_key):
        redis_server_conf = REDIS_SERVER_CONF['servers'][server_key]
        self.pool = redis.ConnectionPool(host = redis_server_conf['HOST'], 
                                        port = redis_server_conf['PORT'], 
                                        db = redis_server_conf['DATABASE'],
                                        decode_responses = True)
        

    @property
    def conn(self):
        if not hasattr(self, '_conn'):
            self.getConnection()
        return self._conn

    def getConnection(self):
        # self._conn = redis.Redis(connection_pool = self.pool, charset= 'utf-8', decode_responses = True)
        self._conn = redis.StrictRedis(connection_pool = self.pool, charset= 'utf-8' )
        return self._conn

    def getPageViewKey(self, user_id):
        return "user:%s:page.view"% user_id

    def getUserInfoKey(self, user_id):
        return "user:%s:info" % user_id

    def getPageView(self, user_id):
        page_view_key = self.getPageViewKey(user_id)
        r = self.getConnection()
        page_view_list = r.lrange(page_view_key,0,-1)
        page_view_list_dto = []
        for pv in page_view_list:
            page_view_dto = PageView.parse_from_string(pv)
            page_view_list_dto.append(page_view_dto)
        return page_view_list_dto

    def getUserInfo(self, user_id):
        user_info_key = self.getUserInfoKey(user_id)
        r = self.getConnection()
        return Session.parse_from_string(r.get(user_info_key))
        
from typing import NamedTuple
class PageView(NamedTuple):
    url: str
    timeNow: int
    timeOnSite: int
    timeOnRead: int

    @classmethod
    def parse_from_string(cls, concate_string):
        if concate_string is not None:
            url, timeNow, timeOnSite, timeOnRead = concate_string.split('\t')
            return cls(url,int(timeNow),int(timeOnSite),int(timeOnRead))
class Session(NamedTuple):
    guid: str
    locId: int
    timeNow: int
    osCode: int
    
    @classmethod
    def parse_from_string(cls, concate_string):
        if concate_string is not None:
            guid, locId, timeNow, osCode = concate_string.split('\t')
            return cls(guid, int(locId), int(timeNow), int(osCode))

# a= PageView.parse_from_string("m.cafebiz.vn/lao-nong-trong-chuoi-kiem-10-ty-nam-sam-oto-sang-chanh-tham-vuon-cho-tien-20190906160523375.chn\t1567828346\t46\t45")
# print(a)
redisClient = RedisClient('localhost')
#for i in range(0,1):
#    r = redisClient.getConnection()
#    page_view_list = redisClient.getPageView("2265891616712405988")
#    print(page_view_list)
#    user_info = redisClient.getUserInfo("2265891616712405988")
#    print(user_info)
# #     # print(r.get('a'))
# print(redisClient.getPageViewKey("ahihi"))
