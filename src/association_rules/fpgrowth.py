import pyfpgrowth

class FpgrowthModel:
    def __init__(self, min_support=2, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        patterns = pyfpgrowth.find_frequent_patterns(transactions, self.min_support)
        rules = pyfpgrowth.generate_association_rules(patterns,  self.min_confidence)
        item_sim_sets = {}
        for key, value in rules.items():
            rule_items = key + value[0]
            for item in rule_items:
                item_sim_sets.setdefault(item, set())
                for i in range(len(rule_items)):
                    if rule_items[i] == item:
                        continue
                    item_sim_sets[item].add(rule_items[i])

        return item_sim_sets

if __name__ == '__main__':
    transactions = [['eggs', 'bacon', 'soup'],
                    ['eggs', 'bacon', 'apple'],
                    ['soup', 'bacon', 'banana']]
    fm = FpgrowthModel()
    item_sim_sets = fm.fit(transactions)
    print(item_sim_sets)

