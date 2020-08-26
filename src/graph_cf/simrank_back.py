import numpy as np
from numpy import matrix

with open('test.txt','r') as log_fp:
    logs = [log.strip() for log in log_fp.readlines()]
    # print(logs)
logs_tuple = [tuple(log.split(",")) for log in logs]
# print (logs_tuple)

queries = list(set([log[0] for log in logs_tuple]))
# print(queries)    #['digital camera', 'flower', 'pc', 'camera', 'tv']
ads = list(set([log[1] for log in logs_tuple]))
# print(ads)#['hp.com', 'teleflora.com', 'bestbuy.com', 'orchids.com']

graph = np.matrix(np.zeros([len(queries),len(ads)]))
# print(graph)   #6行4列的0矩阵

for log in logs_tuple:
    query = log[0]
    ad = log[1]
    q_i = queries.index(query)
    a_j = ads.index(ad)
    graph[q_i,a_j] +=1
print(graph)

query_sim = matrix(np.identity(len(queries)))
print(query_sim)
ad_sim = matrix(np.identity(len(ads)))
print(ad_sim)

def get_ads_num(query):
    q_i = queries.index(query)

    return graph[q_i]

def get_queries_num(ad):
    a_j = ads.index(ad)
    return graph.transpose()[a_j]

def get_ads(query):
    series = get_ads_num(query).tolist()[0]

    return [ads[x] for x in range(len(series)) if series[x] > 0]

def get_queries(ad):
    series = get_queries_num(ad).tolist()[0]
    return [queries[x] for x in range(len(series)) if series[x] > 0]


def query_simrank(q1,q2,c):
    if q1 == q2 :
        return 1
    prefix = c/(get_ads_num(q1).sum() *get_ads_num(q2).sum())
    postfix = 0
    for ad_i in get_ads(q1):
        for ad_j in get_ads(q2):
            i = ads.index(ad_i)
            j = ads.index(ad_j)
            postfix += ad_sim[i,j]
    return prefix*postfix


def ad_simrank(a1,a2,c):
    if a1 == a2 :
        return 1
    prefix = c/(get_queries_num(a1).sum()*get_queries_num(a2).sum())
    postfix = 0
    for query_i in get_queries(a1):
        for query_j in get_queries(a2):
            i = queries.index(query_i)
            j = queries.index(query_j)
            postfix += query_sim[i,j]
    return prefix*postfix


def simrank(c=0.8, times=1):
    global query_sim, ad_sim

    for run in range(times):
        new_query_sim = matrix(np.identity(len(queries)))
        for qi in queries:
            for qj in queries:
                i = queries.index(qi)
                j = queries.index(qj)
                new_query_sim[i,j] =query_simrank(qi,qj,c)

        new_ad_sim = matrix(np.identity(len(ads)))
        for ai in ads:
            for aj in ads:
                i = ads.index(ai)
                j = ads.index(aj)
                new_ad_sim[i,j] =ad_simrank(ai,aj,c)

        query_sim = new_query_sim
        ad_sim = new_ad_sim


if __name__ == '__main__':
    print(queries)
    print(ads)
    simrank()
    print(query_sim)
    print(ad_sim)

