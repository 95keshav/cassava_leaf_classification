from PIL import Image
import pandas as pd
import numpy as np

print("Imported")

base_path = "C:\\Users\\91991\\Documents\\ml_project\\"
csv = pd.read_csv(f"{base_path}train.csv")
print(csv.head())

image_count = 100
x_data = {}
y_data = {}

for index, row in csv.iterrows():
    # print(f"Index: {index} ImageId: {row['image_id']}, label: {row['label']}")
    image = Image.open(f"{base_path}train_images\\{row['image_id']}")
    img_arr = np.array(image)
    img_arr = img_arr.astype("float32")
    img_arr /= 255
    x_data[index] = img_arr
    y_data[index] = row["label"]
    # print(x_data)
    # print(y_data)
    if index >= image_count:
        # breaking after the given limit
        break

x_data = np.array([list(item) for item in x_data.values()])
y_data = np.array([item for item in y_data.values()])
print(x_data.shape)
print(y_data.shape)
