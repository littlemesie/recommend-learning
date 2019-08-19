# -*- coding:utf-8 -*-
import time
import pandas as pd
from profile_server import es_conn

"""
字段	类型	描述
user_id	int	用户 id
pred_gender	string	预测性别 eg : M(男性), F(女性)
pred_age_level	string	预测年龄段, eg: [35,39] 代表年龄位于35到39岁之间
pred_education_degree	int	预测教育程度
pred_career_type	int	预测职业
predict_income	float	预测收入
pred_stage	string	预测人生阶段。 每个人生阶段有一个独特的数字，比如婚姻中代表3，学生状态代表4，已育代表5，那么此字段为 3,4,5
"""

path = '/Volumes/d/CIKM/round2/user_feature.csv'

def build_profile():
    user_feature = pd.read_csv(path, sep='\t', header=None,
                               names=['user_id', 'pred_gender','pred_age_level','pred_education_degree',
                                      'pred_career_type', 'predict_income', 'pred_stage'])
    user_feature.fillna(0, inplace=True)
    es = es_conn.ESSearch()
    for index, row in user_feature.iterrows():

        user_id = row['user_id']
        print(user_id)
        body = {
            "user_info": {
                "pred_age_level": row['pred_age_level'] if row['pred_age_level'] else '',
                "pred_education_degree": row['pred_education_degree'] if row['pred_education_degree'] else 0.0,
                "pred_career_type": row['pred_career_type'] if row['pred_career_type'] else 0.0,
                "predict_income": row['predict_income'] if row['predict_income'] else 0.0,
                "pred_stage": row['pred_stage'] if row['pred_stage'] else '',

            }
        }
        print(body)
        # 插入
        result = es.create(index="user_id", doc_type="user_profile", body=body,id=user_id)


if __name__ == '__main__':
    build_profile()