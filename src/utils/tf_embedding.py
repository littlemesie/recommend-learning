"""
多值离散特征的embedding解决方案
"""
import tensorflow as tf
csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]

TAG_SET = ["harden", "james", "curry", "durant", "paul", "towns", "wrestbrook"]

def sparse_from_csv(csv):
    """
    indices是数组中非0元素的下标，values跟indices一一对应，表示该下标位置的值，最后一个表示的是数组的大小
    :param csv:
    :return:
    """
    ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
    # 这里构造了个查找表
    table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1)
    split_tags = tf.string_split(post_tags_str, "|")

    # 这里给出了不同值通过表查到的index
    return tf.SparseTensor(indices=split_tags.indices, values=table.lookup(split_tags.values),
        dense_shape=split_tags.dense_shape)

# embedding的大小为3
TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))

tags = sparse_from_csv(csv)
# 得到embedding值
# sp_ids就是我们刚刚得到的SparseTensor，而sp_weights=None代表的每一个取值的权重，如果是None的话，所有权重都是1，也就是相当于取了平均。
# 如果不是None的话，我们需要同样传入一个SparseTensor，代表不同球员的喜欢权重。
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)

with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run([embedded_tags]))