import json


UNK = '[UNK]'
PAD = '[PAD]'
MAX_SEQ_LEN = 10


class Processor:
    """数据处理"""
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path

        self.vocab_map = self.load_vocab()
        self.nwords = len(self.vocab_map)

    def load_vocab(self):
        """加载vocab数据"""
        word_dict = {}
        with open(self.vocab_path, encoding='utf8') as f:
            for idx, word in enumerate(f.readlines()):
                word = word.strip()
                word_dict[word] = idx

        return word_dict

    def convert_word2id(self, query):
        """"""
        ids = []
        for w in query:
            if w in self.vocab_map:
                ids.append(self.vocab_map[w])
            else:
                ids.append(self.vocab_map[UNK])
        # 补齐
        while len(ids) < MAX_SEQ_LEN:
            ids.append(self.vocab_map[PAD])

        return ids[:MAX_SEQ_LEN]

    def get_data(self, file_path):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, pos sample, 4 neg sample]], shape = [n, 7]
        """
        data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': [], 'label': []}
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, query_pred, title, tag, label = spline
                if label == '0':
                    continue
                label = int(label)
                cur_arr, cur_len = [], []
                query_pred = json.loads(query_pred)

                # only 4 negative sample
                for each in query_pred:
                    if each == title:
                        continue
                    cur_arr.append(self.convert_word2id(each))
                    each_len = len(each) if len(each) < MAX_SEQ_LEN else MAX_SEQ_LEN
                    cur_len.append(each_len)
                if len(cur_arr) >= 4:
                    data_map['query'].append(self.convert_word2id(prefix))
                    data_map['query_len'].append(len(prefix) if len(prefix) < MAX_SEQ_LEN else MAX_SEQ_LEN)
                    data_map['doc_pos'].append(self.convert_word2id(title))
                    data_map['doc_pos_len'].append(len(title) if len(title) < MAX_SEQ_LEN else MAX_SEQ_LEN)
                    data_map['doc_neg'].extend(cur_arr[:4])
                    data_map['doc_neg_len'].extend(cur_len[:4])
                    data_map['label'].append(label)
                    # data_map['doc_neg'].extend(cur_arr)
                    # data_map['doc_neg_len'].extend(cur_len)
                pass

        return data_map