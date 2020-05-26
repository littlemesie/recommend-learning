from als.als_model import ALS
from utils.movielen_read import read_rating_data
from utils.util import run_time
from utils import metric


def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)


@run_time
def main():
    print("Tesing the performance of ALS...")
    # Load data
    train, test, user_set, item_set = read_rating_data(train_rate=0.7)

    # 得到测试集用户与其所有有正反馈物品集合的映射
    test_user_items = dict()
    test_uids = set()
    for user, item, _ in test:
        test_uids.add(user)
        if user not in test_user_items:
            test_user_items[user] = set()
        test_user_items[user].add(item)
    test_uids = list(test_uids)

    item_popularity = dict()
    for user, item, _ in train:
        if item in item_popularity:
            item_popularity[item] += 1
        else:
            item_popularity.setdefault(item, 1)

    # Train model
    model = ALS()
    model.fit(train, k=3, max_iter=10)

    print("Showing the predictions of users...")
    # Predictions
    predictions = model.predict(test_uids, n_items=10)

    # user_ids = range(1, 5)
    # predictions = model.predict(user_ids, n_items=2)
    recommed_dict = {}
    for user_id, prediction in zip(test_uids, predictions):
        recommed_dict.setdefault(user_id, list())
        for item_id, score in prediction:
            recommed_dict[user_id].append(item_id)
    precision = metric.precision(recommed_dict, test_user_items)
    recall = metric.recall(recommed_dict, test_user_items)
    coverage = metric.coverage(recommed_dict, item_set)
    popularity = metric.popularity(item_popularity, recommed_dict)

    print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage, popularity))


if __name__ == "__main__":
    main()
    # precision:0.3552, recall:0.1117, coverage:0.0719, popularity:5.5114