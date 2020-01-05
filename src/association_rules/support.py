
def load_data():
    """加载数据"""
    data = []
    with open("buy.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = list(map(int, line.replace("\n", "").split(',')))
            data.append(arr)
    return data

def calc_support():
    """计算support"""
    data = load_data()
    sum = len(data)

    itmes = dict()

    for order in data:
        for item in order:
            itmes.setdefault(item, dict())




if __name__ == '__main__':
    data = load_data()
    print(data)