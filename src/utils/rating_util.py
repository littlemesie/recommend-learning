import os
import re
from utils.movielen_read import loadfile

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"


def remove_punctuation(text):
    punctuation = '!,;:?"\'、，；()'
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    return text.strip()

def create_item_dict():
    """生成item 字典"""
    item_dict = {}
    for line in loadfile(base_path + "ml-1m/movies.dat", encoding="ISO-8859-1"):
        arr = line.split("::")
        item_dict.setdefault(arr[0], {})
        title = remove_punctuation(arr[1])
        genres = ' '.join(g.strip('\n') for g in arr[2].split("|"))
        item_dict[arr[0]]['title'] = title
        item_dict[arr[0]]['genres'] = genres

    return item_dict


def create_user_item_dict():
    """生成user-item 字典"""
    item_dict = create_item_dict()
    user_item_dict = {}
    for line in loadfile(base_path + "ml-1m/ratings.dat", encoding="ISO-8859-1"):
        arr = line.split("::")
        user_item_dict.setdefault(arr[0], list())
        movie = item_dict.get(arr[1])
        user_item_dict.get(arr[0]).append(movie)

    return user_item_dict


if __name__ == '__main__':
    create_user_item_dict()