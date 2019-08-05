import xlearn as xl

fm_model = xl.create_fm()
fm_model.setTrain("/Volumes/d/python/xlearn/demo/classification/criteo_ctr/small_train.txt")  # Training data
fm_model.setValidate("/Volumes/d/python/xlearn/demo/classification/criteo_ctr/small_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {
    'task':'binary',
    'lr':0.2,
    'lambda':0.002,
    'metric':'acc'
}

# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, './model.out')

# Prediction task
fm_model.setTest("/Volumes/d/python/xlearn/demo/classification/criteo_ctr/small_test.txt")  # Test data
fm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
fm_model.predict("./model.out", "./output.txt")

class FM():
    """FM推荐"""

    def __init__(self):
        pass