import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from cf.als import ALS
from utils.movielen_read import read_rating_data
from utils.util import run_time


def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)


@run_time
def main():
    print("Tesing the performance of ALS...")
    # Load data
    train, test = read_rating_data()
    # Train model
    model = ALS()
    model.fit(train, k=3, max_iter=10)
    print()

    print("Showing the predictions of users...")
    # Predictions
    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = [format_prediction(item_id, score)
                       for item_id, score in prediction]
        print("User id:%d recommedation: %s" % (user_id, _prediction))


if __name__ == "__main__":
    main()