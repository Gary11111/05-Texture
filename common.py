# -*- coding: utf-8 -*-
import os

class Config:
    '''where to write all the logging information during training(includes saved models)'''
    log_dir = './train_log'

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    exp_name = os.path.basename(log_dir)
    nr_channel = 3
    nr_epoch = 5000
    '''save the image every 10 epoch'''
    save_interval = 100
    '''show the training loss every 10 epoch'''
    show_interval = 10

config = Config()
