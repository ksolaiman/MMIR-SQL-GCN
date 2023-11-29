import pickle
import scipy
import numpy as np
from set_train_test_data_from_mmir_ground import prepare_dataset

print("ad")
# dist will change based on what model we test  - SimGNN, SDML or SQL.
# ground_dist will always remain same
# subset of dist is selected in the fx_calc_map_label() function, that is to select subset of items for image/text/video subset test
np.random.seed(1234)

rows = prepare_dataset()
texttestset = [row[5] for row in rows if row[5]]
imagetestset = [row[3] for row in rows if row[3]]
videotestset = [row[4] for row in rows if row[4]]
print(len(texttestset))
print(len(imagetestset))
print(len(videotestset))

texttrainset = [row[8] for row in rows if row[8]]
imagetrainset = [row[6] for row in rows if row[6]]
videotrainset = [row[7] for row in rows if row[7]]
print(len(texttrainset))
print(len(imagetrainset))
print(len(videotrainset))
textvalidset = [row[11] for row in rows if row[11]]
imagevalidset = [row[9] for row in rows if row[9]]
videovalidset = [row[10] for row in rows if row[10]]
print(len(textvalidset))
print(len(imagevalidset))
print(len(videovalidset))