import env
import pandas as pd
import numpy as np
from PIL import Image
from keras.utils import np_utils
from matplotlib import pyplot as plt

class ImagePreProcessing:
    
    def __init__(self,csv_path,img_limit):
        try:
            if csv_path:
                self.csv = pd.read_csv(csv_path)
                self.x_data = []
                self.y_data = []
                if type(img_limit).__name__ == 'int':
                    self.img_limit = int(img_limit)
                else:
                    self.img_limit = self.csv.shape[0]
        except Exception as e:
            print(e)
    
    def processing(self,image_path,size=(224,224)):
        for index, row in self.csv.iterrows():
            image = Image.open(f"{image_path}{row['image_id']}")
            image = image.resize(size)
            img_arr = np.array(image,dtype='float32')
            img_arr /= 255
            self.x_data.append(img_arr)
            self.y_data.append(row['label'])
            print(index)
            if index >= self.img_limit:
                break
            
    def save_data(self,feat_dest,target_dest):
        np.save(feat_dest,np.array(self.x_data))
        n_classes = 5
        self.y_data = np_utils.to_categorical(self.y_data, n_classes)
        np.save(target_dest,np.array(self.y_data))
        
    def process(self,image_path,feat_dest,target_dest,size=(224,224)):
        self.processing(image_path,size)
        self.save_data(feat_dest,target_dest)
        

base_path = env.BASEPATH
image_limit = str(env.IMAGE_COUNT)
csv_path = env.CSV
image_path = f"{base_path}train_images/"
feat_dest = env.FEATURES_PATH
target_dest = env.LABELS_PATH

preprocessing = ImagePreProcessing(csv_path,999)
preprocessing.process(image_path,feat_dest,target_dest)

x_features = np.load(env.FEATURES_PATH)
y_labels = np.load(env.LABELS_PATH)

plt.imshow(x_features[3])
plt.show()
print(x_features[3].shape)
