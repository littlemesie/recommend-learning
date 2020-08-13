from efficient_apriori import apriori

class AprioriModel:
    def __init__(self, min_support=0.5, min_confidence=1):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        item_sets, rules = apriori(transactions=transactions, min_support=self.min_support, min_confidence=self.min_confidence)

        item_sim_sets = {}
        for rule in rules:
            rule_items = rule.lhs + rule.rhs
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
    am = AprioriModel()
    item_sim_sets = am.fit(transactions)
    print(item_sim_sets)

