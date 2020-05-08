import pandas as pd
import random

user_info_path = 'data/u.user'
item_info_path = 'data/u.item'
base_path = 'data/ua.base'
test_path = 'data/ua.test'

def loadData():
    user_header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_info = pd.read_csv(user_info_path, sep='|', names=user_header)
    user_info['age'] = pd.cut(user_info['age'], bins=[0,10,20,30,40,50,60,70,80,90,100],
                              labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
    user_id = user_info[['user_id']]
    user_info = user_info.drop(columns=['zip_code'])
    user_info = pd.get_dummies(user_info, columns=['user_id', 'age', 'gender', 'occupation'])
    user_info = pd.concat([user_id, user_info], axis=1)
    print(user_info.shape)

    item_header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                   'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_info = pd.read_csv(item_info_path, sep='|', names=item_header, encoding="ISO-8859-1")
    item_info = item_info.drop(columns=['title', 'video_release_date', 'IMDb_URL'])
    item_id = item_info[['item_id']]
    item_info = pd.get_dummies(item_info, columns=['item_id', 'release_date', 'unknown', 'Action', 'Adventure',
                   'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    item_info = pd.concat([item_id, item_info], axis=1)
    print(item_info.shape)

    header = ['user_id', 'item_id', 'rating', 'timestamp']
    base = pd.read_csv(base_path, sep='\t', names=header)
    base = base.drop(columns=['timestamp'])
    base = pd.merge(base, user_info, how='left', on='user_id')
    base = pd.merge(base, item_info, how='left', on='item_id')
    base = base.drop(columns=['user_id', 'item_id'])
    print(base.shape)
    # print(base.head())
    # print(base.dtypes)
    test = pd.read_csv(test_path, sep='\t', names=header)
    test = test.drop(columns=['timestamp'])
    test = pd.merge(test, user_info, how='left', on='user_id')
    test = pd.merge(test, item_info, how='left', on='item_id')
    test = test.drop(columns=['user_id', 'item_id'])
    print(test.shape)
    # print(test.head())
    # print(test.dtypes)
    return base, test

def shuffleBatch(x_batch, y_batch):
    assert len(x_batch) == len(y_batch)
    length = len(x_batch)
    index = [i for i in range(length)]
    random.shuffle(index)
    x_batch_shuffle = [x_batch[i] for i in index]
    y_batch_shuffle = [y_batch[i] for i in index]
    return x_batch_shuffle, y_batch_shuffle

def getBatchData(data, batch_size=32):
    rating = data.rating
    data = data.drop(columns=['rating'])
    start, end = 0, 0
    while True:
        start = end % data.shape[0]
        end = min(data.shape[0], start + batch_size)
        x_batch, y_batch = [], []
        for i in range(start, end):
            label = 1 if rating.iloc[i] >= 4 else 0
            y_batch.append(label)
            single_sample = data.iloc[i, :].values
            x_batch.append(single_sample)

        x_batch_shuffle, y_batch_shuffle = shuffleBatch(x_batch, y_batch)
        yield x_batch_shuffle, y_batch_shuffle

# loadData()