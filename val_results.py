
from __future__ import print_function
import collections
from bresenham import bresenham
import argparse
import os
import numpy as np
from math import log10
from matplotlib import pyplot as plt
#from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#import matplotlib.pyplot as plt
from torchvision.utils import save_image

from datasets import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate
from PIL import Image

cudnn.benchmark = True


def check_connection(img):
    indent = 0
    success = True
    #new_prediction = img
    # print(np.unique(img.float().round().detach().cpu().numpy()))
    inds = np.where(
        np.array(img.float().round().detach().tolist()).astype('uint8') == 0)
    if len(inds) < 0:
        return False, 1
    prev_x, prev_y = sorted(zip(inds[1], inds[0]))[0]
    for x, y in sorted(zip(inds[1], inds[0]))[1:]:
        if abs(prev_x - x) <= 1 and abs(prev_y - y) <= 1:
            prev_x = x
            prev_y = y
            continue
        if prev_x == x:
            prev_y = y
            continue

        indent += 1
       # print(list(bresenham(prev_x, prev_y, x, y)))
        for cell in list(bresenham(prev_x, prev_y, x, y))[1:-1]:
            print("bresenham")
            print(img[cell[1], cell[0]])
            if (img[cell[1], cell[0]] == 1).all():
                success = False
            else:
                img[cell[1], cell[0]] = 0
        prev_x = x
        prev_y = y

    return success, indent


img_size = 64
channels = 1
num_classes = 3
pred_folder = './predictions/size_64/20_den2/20_den_25e/'
result_folder = './results/size_64/20_den_25e'
dataset_dir = './data/size_64/20_den2'
device = torch.device("cpu")

dataset = ImageDataset(dataset_dir, img_size=img_size)
print(f"Dataset size: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError(
        "Dataset is empty! Check the dataset directory and image files.")

os.makedirs(result_folder, exist_ok=True)

result_folders = ['./results/size_64/20_den_50e/']

models = [define_G(channels, num_classes, 64, 'batch', False,
                   'normal', 0.02, gpu_id=device, use_ce=True, ce=False, unet=False)]

for model, path in zip(models, result_folders):
    model.load_state_dict(torch.load(path + '/generator.pt'))

batch_size = 6

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=1, shuffle=False, num_workers=0)

#criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
#criterionCE = nn.CrossEntropyLoss().to(device)
avg_psnr = [0] * len(models)
number_of_indents = [0] * len(models)
success_rate = [0] * len(models)
'''
for i, batch in enumerate(val_data_loader):
    #if i > 10: break
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        
        #print(np.unique(prediction.round().detach().cpu().numpy()))
        mse = criterionMSE(prediction.float(), target.float())
        #psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += mse
        success, indent, _ = check_connection(prediction.detach().cpu())
        number_of_indents[num] += indent
        success_rate[num] += success
    if i > 10:
        break
        #if i % 50 == 0:
        #    predictions += [prediction.float().data]
    #if i % 100  == 0:
       # sample = torch.cat((input.data, target.data, *predictions))
       # save_image(torch.transpose(sample, 0, 1), result_folder  + ('%d.png' % i), nrow=1, normalize=True, pad_value=0)
print(i, number_of_indents, success_rate)
with open(result_folder + 'val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])
    f.writelines(["%s " % (item / i)  for item in number_of_indents])
    f.writelines(["%s " % (item / i)  for item in success_rate])

'''
# for i, batch in enumerate(val_data_loader):
#     input, target = batch[0].to(device), batch[1].to(device)
#     predictions = []
#     for num, model in enumerate(models):
#         if num == 0:
#             input_ = input - torch.ones_like(input)
#             prediction_ = model(input_)
#             prediction = prediction_ + torch.ones_like(prediction_)
#         else:  # ignored with only one model
#             output = model(input)
#             _, prediction = torch.max(output, 1, keepdim=True)
#         prediction = torch.where(target.float() == 1, torch.ones_like(
#             target).float(), prediction.float())
#         predictions += [prediction.float().data]
#         if num == len(models) - 1:  # always true if only one model
#             success, indent = check_connection(prediction.detach().cpu())
#             predictions += [prediction.float().data]
#     # print("input shape:", input.data.shape)
#     # print("target shape:", target.data.shape)
#     for i, pred in enumerate(predictions):
#         print(f"prediction {i} shape:", pred.shape)

#     processed_predictions = []
#     for pred in predictions:
#         pred = pred[:, 0:1, :, :]
#         processed_predictions.append(pred)
#     sample = torch.cat((input.data, target.data, *processed_predictions), 0)
#     print(sample.size())
#     save_image(sample, result_folder + ('%d.png' % i),
#                nrow=7, normalize=True, pad_value=255)
#     if i > 10:
#         break


# for i, batch in enumerate(val_data_loader):
#     input, target = batch[0].to(device), batch[1].to(device)
#     predictions = []
#     for num, model in enumerate(models):
#         if num == 0:
#             input_ = input - torch.ones_like(input)
#             prediction_ = model(input_)
#             prediction = prediction_ + torch.ones_like(prediction_)
#         else:
#             output = model(input)
#             _, prediction = torch.max(output, 1, keepdim=True)
#         prediction = torch.where(target.float() == 1, torch.ones_like(
#             target).float(), prediction.float())
#         predictions += [prediction.float().data]
#         if num == len(models) - 1:
#             success, indent = check_connection(prediction.detach().cpu())
#             predictions += [prediction.float().data]

#     processed_predictions = []
#     for pred in predictions:
#         pred = pred[:, 0:1, :, :]
#         processed_predictions.append(pred)

#     sample = torch.cat((input.data, target.data, *processed_predictions), 0)
#     pred_img = processed_predictions[-1]

#     sample_min = sample.min()
#     sample_max = sample.max()
#     sample = (sample - sample_min) / (sample_max - sample_min)
#     sample = sample.clamp(0, 1)

#     pred_min = pred_img.min()
#     pred_max = pred_img.max()
#     pred_img = (pred_img - pred_min) / (pred_max - pred_min)
#     pred_img = pred_img.clamp(0, 1)

#     save_image(sample, pred_folder + ('%d.png' % i),
#                nrow=7, normalize=False, pad_value=0)

#     save_image(pred_img, pred_folder + ('prediction_%d.png' % i),
#                nrow=7, normalize=False, pad_value=0)

#     print(f'Processed batch {i+1}/{len(val_data_loader)}')


for i, batch in enumerate(val_data_loader):
    input = batch[0].to(device)
    predictions = []
    for num, model in enumerate(models):
        output = model(input)
        m, prediction = torch.max(output, 1, keepdim=True)
        print(m)
        predictions += [prediction.float().data]
        if num == len(models) - 1:
            success, indent = check_connection(prediction.detach().cpu())
            predictions += [prediction.float().data]

    processed_predictions = []
    for pred in predictions:
        pred = pred[:, 0:1, :, :]
        processed_predictions.append(pred)

    sample = torch.cat((input.data, *processed_predictions), 0)
    pred_img = processed_predictions[-1]

    sample_min = sample.min()
    sample_max = sample.max()
    sample = (sample - sample_min) / (sample_max - sample_min)
    sample = sample.clamp(0, 1)

    pred_min = pred_img.min()
    pred_max = pred_img.max()
    pred_img = (pred_img - pred_min) / (pred_max - pred_min)
    pred_img = pred_img.clamp(0, 1)

    save_image(sample, pred_folder + ('%d.png' % i),
               nrow=7, normalize=False, pad_value=0)

    save_image(pred_img, pred_folder + ('prediction_%d.png' % i),
               nrow=7, normalize=False, pad_value=0)

    print(f'Processed batch {i+1}/{len(val_data_loader)}')


'''
dataset_dir = './size_64/all_den/'

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=6, shuffle=True, num_workers=1)
avg_psnr = [0] * len(models)

for i, batch in enumerate(val_data_loader):
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        mse = criterionMSE(prediction.float(), target.float())
        psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += psnr
        if i % 450 == 0:
            predictions += [prediction.float().data]
    if i % 450 == 0:
        sample = torch.cat((input.data, target.data, *predictions), -1)
        save_image(sample, result_folder  + 'all_' + ('%d.png' % i), nrow=1, normalize=True)
with open(result_folder + 'all_val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])

dataset_dir = './size_64/round/'

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=6, shuffle=True, num_workers=1)
avg_psnr = [0] * len(models)

for i, batch in enumerate(val_data_loader):
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        mse = criterionMSE(prediction.float(), target.float())
        psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += psnr
        predictions += [prediction.float().data]
    if i % 450 == 0:
        sample = torch.cat((input.data, target.data, *predictions), -1)
        save_image(sample, result_folder  + 'round_' + ('%d.png' % i), nrow=1, normalize=True)
with open(result_folder + 'round_val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])
'''
