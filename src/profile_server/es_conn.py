# encoding: utf-8
from elasticsearch import Elasticsearch

class ESSearch:

    def __init__(self):
        self.es_conn = Elasticsearch(host='127.0.0.1', port=9200, timeout=60, max_retries=10, retry_on_timeout=True)

    def conn(self):
        return self.es_conn

    def ping(self, **query_params):
        return self.es_conn.ping(**query_params)

    def info(self, **query_params):
        return self.es_conn.info(**query_params)

    def create(self, index, doc_type, body, id, **query_params, ):
        """
        build create
        :param index:
        :param doc_type:
        :param id:
        :param body:
        :param query_params:
        :return:
        """
        return self.es_conn.create(index=index, doc_type=doc_type, id=id, body=body, **query_params)

    def index(self, index, doc_type, body, id=None, **query_params):
        """
        build index
        :param index:
        :param doc_type:
        :param body:
        :param id:
        :param query_params:
        :return:
        """
        return self.es_conn.index(index=index, doc_type=doc_type, body=body, id=id, **query_params)

    def search(self, index, doc_type, body, params=None):
        """
        build search
        :param index: index name
        :param doc_type: doc_type
        :param body: body
        :return: search result
        """
        if params:
            return self.es_conn.search(index=index, doc_type=doc_type, body=body, params=params)

        return self.es_conn.search(index=index, doc_type=doc_type, body=body)

    def get(self, index, doc_type, id, params=None):
        """
        get
        :param index:
        :param doc_type:
        :param id:
        :param params:
        :return:
        """
        return self.es_conn.get(index=index, doc_type=doc_type, id=id, params=params)

    def update(self, index, doc_type, id, body=None, params=None):
        """
        update
        :param index:
        :param doc_type:
        :param id:
        :param body:
        :param params:
        :return:
        """
        return self.es_conn.update(index=index, doc_type=doc_type, id=id, body=body, params=params)

    def delete(self, index, doc_type, id, params=None):
        """
        delete
        :param index:
        :param doc_type:
        :param id:
        :return:
        """
        return self.es_conn.delete(index=index, doc_type=doc_type, id=id, params=params)

    def build_index_body(self, key, value):
        """构建插入一个字段的body"""
        body = {
            "size": 0,
            key: {
                "value": value
            }
        }
        return body

    def key_word_search_body(self, key, value):
        """value:1关键词查询,2也可以是短语全文搜索"""
        body = {
            "query": {
                "match": {
                    key: value
                }
            }
        }
        return body

    def range_search_body(self, key, gt_value, lt_value):
        """范围查询"""
        body = {
            "query": {
                "range": {
                    key: {"gt": gt_value, "lt": lt_value}
                }

            }
        }
        return body

    def match_phrase_search_body(self, key, value):
        """短语查询"""
        body = {
            "query": {
                "match_phrase": {
                    key: value
                }

            },
            "highlight": {
                "fields": {
                    key: {}
                }

            }
        }
        return body

    def aggs_search_body(self, key, aggregations):
        """聚合"""
        body = {
            "aggs": {
                aggregations: {
                    "terms": {'field': key}
                }

            }
        }
        return body


if __name__ == '__main__':
    """简单的使用"""
    es = ESSearch()
    stu_no = 123456
    # es_score_body = es.build_index_body('stu_no',stu_no)
    # # 插入
    # result = es.index(index="stuscore", doc_type="logs", body=es_score_body)
    # 查询所有
    # result = es.search(index="stuscore", doc_type="logs", body='')
    # 获取对应的值
    # scores = []
    # for hit in result['hits']['hits']:
    #     scores.append(str(hit['_source']['stu_no']['value']))
    # 查询100条
    # result = es.search(index="bank", doc_type="account", body='', params={'size':100})
    # 关键词查询
    # key_word_search = es.key_word_search_body('firstname','Ratliff')
    # result = es.search(index="bank", doc_type="account", body=key_word_search)
    # 范围查询
    # filter_search = es.range_search_body('age', 20, 30)
    # result = es.search(index="bank", doc_type="account", body=filter_search)
    # 短语查询
    # match_phrase_search = es.key_word_search_body('address','National Drive')
    # result = es.search(index="bank", doc_type="account", body=match_phrase_search)
    # 聚合
    aggs_search_body = es.aggs_search_body('age','ages')
    result = es.search(index="bank", doc_type="account", body=aggs_search_body)
    print(result)
    # for hit in result['hits']['hits']:
    #     print(hit)


