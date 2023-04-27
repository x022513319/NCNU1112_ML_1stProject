import os
from PIL import Image
import numpy as np

#讀取資料夾212張圖片，圖片為彩色圖，3通道，圖像大小：150*250
def load_data():
    data_1 = np.empty((203, 3, 150, 250),dtype="uint8")
    label_1 = np.empty((203,),dtype="uint8")
    data_2 = np.empty((10, 3, 150, 250),dtype="uint8")
    label_2 = np.empty((10,),dtype="uint8")

    imgs_1 = os.listdir("./trainimg")
    num_1 = len(imgs_1)
    for i in range(num_1):
        img_1 = Image.open("./trainimg/"+imgs_1[i])
        arr_1 = np.asarray(img_1)
        data_1[i,:,:,:] = [arr_1[:,:,0],arr_1[:,:,1],arr_1[:,:,2]]
        label_1[i] = int(imgs_1[i].split('.')[0])

    imgs_2 = os.listdir("./testimg")
    num_2 = len(imgs_2)
    for i in range(num_2):
        img_2 = Image.open("./testimg/"+imgs_2[i])
        arr_2 = np.asarray(img_2)
        data_2[i,:,:,:] = [arr_2[:,:,0],arr_2[:,:,1],arr_2[:,:,2]]
        label_2[i] = int(imgs_2[i].split('.')[0])
    
    return (data_1,label_1), (data_2,label_2)
