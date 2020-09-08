import numpy as np
import faiss

# 向量个数
num_vec = 5000
# 向量维度
vec_dim = 768
# 搜索topk
topk = 10

# 随机生成一批向量数据
vectors = np.random.rand(num_vec, vec_dim).astype('float32')
print(vectors.shape)

# 创建索引
faiss_index = faiss.IndexFlatL2(vec_dim)  # 使用欧式距离作为度量
# 添加数据
faiss_index.add(vectors)

# 查询向量 假设有5个
query_vectors = np.random.rand(5, vec_dim).astype('float32')
print(query_vectors)
# 搜索结果
# 分别是 每条记录对应topk的距离和索引
# ndarray类型 。shape：len(query_vectors)*topk
res_distance, res_index = faiss_index.search(query_vectors, topk)
print(res_index)
print(res_distance)