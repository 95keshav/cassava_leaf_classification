import env
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

base_path = env.BASEPATH
csv = pd.read_csv(f"{base_path}{env.CSV}")
x_data = []
y_data = []

for index, row in csv.iterrows():
    # print(f"Index: {index} ImageId: {row['image_id']}, label: {row['label']}")
    image = Image.open(f"{base_path}train_images\\{row['image_id']}")
    img_arr = np.array(image,dtype='float32')
    img_arr /= 255
    x_data.append(img_arr)
    y_data.append(row['label'])
    if index >= env.IMAGE_COUNT:
        break

np.save(env.FEATURES_PATH,np.array(x_data))
x_features = np.load(env.FEATURES_PATH)
np.save(env.LABELS_PATH,np.array(y_data))
y_labels = np.load(env.LABELS_PATH)
plt.imshow(x_features[3])
plt.show()
print(len(y_labels))
print(y_labels)
