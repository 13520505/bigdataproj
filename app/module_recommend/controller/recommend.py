from flask import Blueprint, request, make_response
import app.module_recommend.service.recommend as rec

# Create blueprint
recommend_page = Blueprint('recommend_page', __name__,
                        template_folder='templates')

import time


def timer(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print('total run-time of %r: %f ms' % (fn.__name__, (end_time - start_time) * 1000))
        return result
    return wrapper

#Routing
@recommend_page.route('/ngoc', methods=['POST', 'GET'])
def hello():
    return 'Hello, ng'

@recommend_page.route('/recommend', methods=['POST', 'GET'])
def recommend():
    news_id = request.values.get("news_id")
    user_id = request.values.get("user_id")
    return rec.recommendSR(news_id, user_id)


