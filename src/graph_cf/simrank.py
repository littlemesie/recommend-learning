import numpy as np

class Simrank:
    """Simrank的协同过滤"""

    def __init__(self, c=0.1, epoch=1):
        self.c = c
        self.epoch = epoch
        self.user_set = []
        self.item_set = []
        self.data = []

    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def get_data(self, file_name):
        user_set = set()
        item_set = set()
        for line in self.load_file(file_name):
            user, movie, rating, timestamp = line.split('\t')
            self.data.append([user, movie, rating])
            user_set.add(user)
            item_set.add(movie)
        self.user_set = list(user_set)
        self.item_set = list(item_set)

    def get_user_num(self, user, graph):
        user_i = self.user_set.index(user)
        return graph[user_i]

    def get_item_num(self, item, graph):
        item_j = self.item_set.index(item)
        return graph.transpose()[item_j]

    def get_user(self, user, graph):
        series = self.get_user_num(user, graph).tolist()[0]
        return [self.item_set[x] for x in range(len(series)) if series[x] > 0]

    def get_item(self, item, graph):
        series = self.get_item_num(item, graph).tolist()[0]
        return [self.user_set[x] for x in range(len(series)) if series[x] > 0]

    def user_simrank(self, graph, item_sim, user_1, user_2, c):
        if user_1 == user_2:
            return 1
        prefix = c / (self.get_user_num(user_1, graph).sum() * self.get_user_num(user_2, graph).sum())
        postfix = 0
        for user_i in self.get_user(user_1, graph):
            for user_j in self.get_user(user_2, graph):
                i = self.item_set.index(user_i)
                j = self.item_set.index(user_j)
                postfix += item_sim[i, j]
        return prefix * postfix

    def item_simrank(self, graph, user_sim, item_1, item_2, c):
        if item_1 == item_2:
            return 1
        prefix = c / (self.get_item_num(item_1, graph).sum() * self.get_item_num(item_2, graph).sum())
        postfix = 0
        for item_i in self.get_item(item_1, graph):
            for item_j in self.get_item(item_2, graph):
                i = self.user_set.index(item_i)
                j = self.user_set.index(item_j)
                postfix += user_sim[i, j]
        return prefix * postfix

    def simrank_calc(self):
        user_sim = np.matrix(np.identity(len(self.user_set)))
        item_sim = np.matrix(np.identity(len(self.item_set)))

        graph = np.matrix(np.zeros([len(self.user_set), len(self.item_set)]))

        for d in self.data:
            user = d[0]
            item = d[1]
            user_i = self.user_set.index(user)
            item_j = self.item_set .index(item)
            graph[user_i, item_j] += 1

        for run in range(self.epoch):
            new_user_sim = np.matrix(np.identity(len(self.user_set)))
            for user_i in self.user_set:
                for user_j in self.user_set:
                    i = self.user_set.index(user_i)
                    j = self.user_set.index(user_j)
                    new_user_sim[i, j] = self.user_simrank(graph, item_sim, user_i, user_j, self.c)

            new_item_sim = np.matrix(np.identity(len(self.item_set)))
            for item_i in self.item_set:
                for item_j in self.item_set:
                    i = self.item_set.index(item_i)
                    j = self.item_set.index(item_i)
                    new_item_sim[i, j] = self.item_simrank(graph, user_sim, item_i, item_j, self.c)

            user_sim = new_user_sim
            item_sim = new_item_sim

        return user_sim, item_sim

if __name__ == '__main__':
    ss = Simrank()
    import os
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    rating_file = base_path + 'ml-100k/u.data'
    ss.get_data(rating_file)
    user_sim, item_sim = ss.simrank_calc()
    print(user_sim)
