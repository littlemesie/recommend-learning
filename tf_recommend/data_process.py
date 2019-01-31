# coding=utf-8
import re
import pickle
import pandas as pd

def users_data():
    """用户数据"""
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('../data/users.dat', sep='::', header=None, names=users_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')

    # 没有做数据处理的原始用户数据
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)
    # 年龄处理后:{1: 0, 35: 1, 45: 2, 50: 3, 18: 4, 56: 5, 25: 6}
    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)
    return users, users_orig

def movies_data():
    """电影数据"""
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('../data/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    movies_orig = movies.values

    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}
    # 将电影类型转成等长数字列表，长度是18(1x18)
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}
    # 将电影Title转成等长数字列表，长度是15(1x15)
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])
    movies['Title'] = movies['Title'].map(title_map)

    return movies, movies_orig, title_count, title_set, genres2int

def ratings_data():
    """读取评分数据集"""
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('../data/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    return ratings

def merge_data():
    """
    合并三个表
    :return:
    title_count：Title字段的长度（15）
    title_set：Title文本的集合
    genres2int：电影类型转数字的字典
    features：是输入X
    targets_values：是学习目标y
    ratings：评分数据集的Pandas对象
    users：用户数据集的Pandas对象
    movies：电影数据的Pandas对象
    data：三个数据集组合在一起的Pandas对象
    movies_orig：没有做数据处理的原始电影数据
    users_orig：没有做数据处理的原始用户数据
    """
    users, users_orig = users_data()
    movies, movies_orig, title_count, title_set, genres2int = movies_data()
    ratings = ratings_data()
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    features = features_pd.values
    targets_values = targets_pd.values
    pickle.dump((
                title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig,
                users_orig), open('preprocess.p', 'wb'))
    # return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


if __name__ == '__main__':
    merge_data()

