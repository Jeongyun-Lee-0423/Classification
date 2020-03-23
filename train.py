# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:35:23 2019

@author: 이정윤
"""
from model import *
from data import *
#from keras.utils import multi_gpu_model


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

if(1):
#with tf.device('/gpu:0'):
    data_gen_args = dict(rotation_range=0,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip = True,
                            fill_mode='nearest')
    
    myGene = trainGenerator(5,'D:/WORKPLACE/make_image/train','image','label',data_gen_args,save_to_dir = None)  
    #myGene = trainGenerator(4,'D:/DL_CODE/DeepLearning/unet-master/data/membrane/train','image','label',data_gen_args,save_to_dir = None)     
    model = unet()    
    model.load_weights('unet_membrane.hdf5')  
    
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=200,epochs=500,callbacks=[model_checkpoint])
    
    
    
   # testGene = testGenerator("D:/WORKPLACE/make_image/test")
   # results = model.predict_generator(testGene,100, verbose=1)
   # saveResult("data/membrane/test",results)