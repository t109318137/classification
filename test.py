import numpy as np
import h5py
import glob

import sklearn.metrics
import tensorflow.keras.optimizers
import os

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization,MaxPool2D
from keras.layers import Conv2D,MaxPooling2D
import cv2

from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping



dict_characters = {0: 'abraham_grampa_simpson',1: 'agnes_skinner',2: 'apu_nahasapeemapetilon',
                   3: 'barney_gumble',4: 'bart_simpson',5: 'brandine_spuckler',
                   6:'carl_carlson',7:'charles_montgomery_burns',8:'chief_wiggum',
                   9:'cletus_spuckler',10:'comic_book_guy',11:'disco_stu',
                   12:'dolph_starbeam',13:'duff_man',14:'edna_krabappel',
                   15:'fat_tony',16:'gary_chalmers',17:'gil',
                   18:'groundskeeper_willie',19:'homer_simpson',20:'jimbo_jones',
                   21:'kearney_zzyzwicz',22:'kent_brockman',23:'krusty_the_clown',
                   24:'lenny_leonard',25:'lionel_hutz',26:'lisa_simpson',
                   27:'lunchlady_doris',28:'maggie_simpson',29:'marge_simpson',
                   30:'martin_prince',31:'mayor_quimby',32:'milhouse_van_houten',
                   33:'miss_hoover',34:'moe_szyslak',35:'ned_flanders',
                   36:'nelson_muntz',37:'otto_mann',38:'patty_bouvier',
                   39:'principal_skinner',40:'professor_john_frink',41:'rainier_wolfcastle',
                   42:'ralph_wiggum',43:'selma_bouvier',44:'sideshow_bob',
                   45:'sideshow_mel',46:'snake_jailbird',47:'timothy_lovejoy',
                   48:'troy_mcclure',49:'waylon_smithers'}

img_width = 60
img_height = 60

num_classes = len(dict_characters)
test_size= 0.15
imgsPath='/tmp/theSimpsons-train/train'


def load_pictures():
    pics = []
    labels = []

    for k, v in dict_characters.items():
        pictures=[k for k in glob.glob(imgsPath + "/" + v + "/*")]
        print(v + " : " + str(len(pictures)))
        for i,pic in enumerate(pictures):
            tmp_img = cv2.imread(pic)


            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img,(img_height,img_width))
            pics.append(tmp_img)
            labels.append(k)
    return np.array(pics),np.array(labels)

def get_dataset(save =False, load = False):
    if(load):
        h5f = h5py.File('train60.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()

        h5f = h5py.File('valid60.h5','r')
        y_train=h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
    else:

        X,y= load_pictures()
        y= keras.utils.np_utils.to_categorical(y,num_classes)

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
        if save:
            h5f=h5py.File('train60.h5','w')
            h5f.create_dataset('X_train',data=X_train)
            h5f.create_dataset('X_test',data=X_test)
            h5f.close()

            h5f = h5py.File('valid60.h5','w')
            h5f.create_dataset('y_train',data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()

    X_train = X_train.astype('float32')/255.
    X_test = X_test.astype('float32')/255.
    print("Train",X_train.shape,y_train.shape)
    print("Valid",X_test.shape,y_test.shape)

    return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=get_dataset(save=True,load=False)
datagen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

def creat_model():
    input_shape = (img_height,img_width,3)
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
    model.add(Conv2D(32,kernel_size=(3,3),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=(3,3),padding='Same',activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(86, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(86, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(1024,activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(50,activation="softmax"))

    optimizer = 'Adam'
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    return model


model=creat_model()
model.summary()
lr=0.001

batch_size = 32
epochs=50
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))



def read_images(path):
    images=[]
    for i in range(10791):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(60,60))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image)
    images=np.array(images,dtype=np.float32)/255
    return images

from keras.models import load_model
model =load_model('/models/model.h5')


from chainercv.utils import read_image

imgsPath = '/tmp/theSimpsons-test/test/'


read_test_images = read_images(imgsPath)
predict = model.predict(read_test_images)
predict = np.argmax(predict,axis=1)

dict_characters = {0: 'abraham_grampa_simpson',1: 'agnes_skinner',2: 'apu_nahasapeemapetilon',
                   3: 'barney_gumble',4: 'bart_simpson',5: 'brandine_spuckler',
                   6:'carl_carlson',7:'charles_montgomery_burns',8:'chief_wiggum',
                   9:'cletus_spuckler',10:'comic_book_guy',11:'disco_stu',
                   12:'dolph_starbeam',13:'duff_man',14:'edna_krabappel',
                   15:'fat_tony',16:'gary_chalmers',17:'gil',
                   18:'groundskeeper_willie',19:'homer_simpson',20:'jimbo_jones',
                   21:'kearney_zzyzwicz',22:'kent_brockman',23:'krusty_the_clown',
                   24:'lenny_leonard',25:'lionel_hutz',26:'lisa_simpson',
                   27:'lunchlady_doris',28:'maggie_simpson',29:'marge_simpson',
                   30:'martin_prince',31:'mayor_quimby',32:'milhouse_van_houten',
                   33:'miss_hoover',34:'moe_szyslak',35:'ned_flanders',
                   36:'nelson_muntz',37:'otto_mann',38:'patty_bouvier',
                   39:'principal_skinner',40:'professor_john_frink',41:'rainier_wolfcastle',
                   42:'ralph_wiggum',43:'selma_bouvier',44:'sideshow_bob',
                   45:'sideshow_mel',46:'snake_jailbird',47:'timothy_lovejoy',
                   48:'troy_mcclure',49:'waylon_smithers'}

#print('\n',sklearn.metrics.classification_report(true,predict,target_name=list(dict_characters.values())),sep='')

with open('predict2.csv','w')as f:
    f.write('id,character\n')
    for i in range(len(predict)):
        f.write(str(i+1)+","+dict_characters[((predict[i]))]+"\n")

