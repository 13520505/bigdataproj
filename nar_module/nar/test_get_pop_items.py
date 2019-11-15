from nar_module.nar.RecentlyPopularRecommender import RecentlyPopularRecommender
import time
def main():
    start = time.time()
    pop = RecentlyPopularRecommender()
    popular_item_ids = pop.get_recent_popular_item_ids()
    print(popular_item_ids[0:100])
    print(str(time.time() - start))


if __name__ == '__main__':
    main()
