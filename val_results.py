
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


# def check_connection(img):
#     indent = 0
#     success = True
#     #new_prediction = img
#     # print(np.unique(img.float().round().detach().cpu().numpy()))
#     inds = np.where(
#         np.array(img.float().round().detach().tolist()).astype('uint8') == 0)
#     if len(inds) < 0:
#         return False, 1
#     prev_x, prev_y = sorted(zip(inds[1], inds[0]))[0]
#     for x, y in sorted(zip(inds[1], inds[0]))[1:]:
#         if abs(prev_x - x) <= 1 and abs(prev_y - y) <= 1:
#             prev_x = x
#             prev_y = y
#             continue
#         if prev_x == x:
#             prev_y = y
#             continue

#         indent += 1
#        # print(list(bresenham(prev_x, prev_y, x, y)))
#         for cell in list(bresenham(prev_x, prev_y, x, y))[1:-1]:
#             print("bresenham")
#             print(img[cell[1], cell[0]])
#             if (img[cell[1], cell[0]] == 1).all():
#                 success = False
#             else:
#                 img[cell[1], cell[0]] = 0
#         prev_x = x
#         prev_y = y

#     return success, indent


def check_connection(out, img):
    # img_size = out.shape[1]
    if len(out.shape) == 4:  # [batch, channel, height, width]
        out = out[0, 0]
    elif len(out.shape) == 3:  # [channel, height, width]
        out = out[0]

    if len(img.shape) == 4:  # [batch, channel, height, width]
        img = img[0, 0]
    elif len(img.shape) == 3:  # [channel, height, width]
        img = img[0]

    img_size = int(np.sqrt(out.numel()))
    grid = out.reshape(img_size, img_size)
    img = img.reshape(img_size, img_size)
    grid[np.array(img.tolist()).astype('uint8') == 0] = 0
    grid[np.array(img.tolist()).astype('uint8') == 1] = 1

    inds = np.where(np.array(grid.reshape(
        img_size, img_size).tolist()).astype('uint8') == 0)

    path_cells = {}
    for x, y in zip(inds[0], inds[1]):
        path_cells[x * img_size + y] = True

    startgoal = np.array(np.where(np.array(img.tolist()).astype('uint8') == 0))
    start_x, start_y = startgoal[:, 0]
    goal_x, goal_y = startgoal[:, 1]

    success = True
    succes_w_bresenham = True
    final_path = []
    length_true = 0

    path_cells[start_x * img_size + start_y] = False
    while not (start_x == goal_x and start_y == goal_y):
        final_path += [(start_x, goal_x)]
        neighbor = False
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            x, y = start_x + dx, start_y + dy
            if x < 0 or y < 0 or x >= img_size or y >= img_size:
                continue
            if x * img_size + y in path_cells and path_cells[x * img_size + y]:
                path_cells[x * img_size + y] = False
                length_true += np.sqrt((start_x - x) ** 2 + (start_y - y) ** 2)
                start_x, start_y = x, y
                neighbor = True
                break
        if neighbor:
            continue

        success = False
        min_dist = np.inf
        closest = None
        for x, y in zip(inds[0], inds[1]):
            if (start_x - x) ** 2 + (start_y - y) ** 2 < min_dist and path_cells[x * img_size + y]:
                min_dist = (start_x - x) ** 2 + (start_y - y) ** 2
                closest = (x, y)
        for cell in list(bresenham(start_x, start_y, closest[0], closest[1]))[1:-1]:
            # print(cell[0], cell[1])
            final_path += [cell]
            if out[cell[0], cell[1]] == 1:
                succes_w_bresenham = False
            else:
                out[cell[0], cell[1]] = 0
        if succes_w_bresenham:
            start_x, start_y = closest
            path_cells[closest[0] * img_size + closest[1]] = False
        else:
            break
    return success, succes_w_bresenham, final_path


img_size = 64
channels = 1
num_classes = 3
pred_folder = './predictions/size_64/ideal/'
result_folder = './results/size_64/20_den3_100e_32b'
dataset_dir = './data/size_64/ideal/'
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

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval', img_size=img_size),
                             batch_size=1, shuffle=False)

#criterionL1 = nn.L1Loss().to(device)
# criterionMSE = nn.MSELoss().to(device)
# criterionCE = nn.CrossEntropyLoss().to(device)
avg_psnr = []
number_of_indents = [0] * len(models)
success_rate = []
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
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        output = model(input)
        # plt.figure()
        # plt.imshow(output[0, 0].detach().cpu().numpy(), cmap='gray')
        # plt.show()
        # plt.close()
        _, prediction = torch.max(output, 1, keepdim=True)
        # plt.figure()
        # plt.imshow(prediction[0, 0].detach().cpu().numpy(), cmap='gray')
        # plt.show()
        # plt.close()
        predictions += [prediction.float().data]
        success, success_w_bresenham, final_path = check_connection(prediction.float().detach().cpu(), input.float().detach().cpu())
        print(f"Path: {final_path}")
        print(f"Success: {success}, Success w/ Bresenham: {success_w_bresenham}")
        success_rate.append(success_w_bresenham)
        predictions += [prediction.float().data]
            
    # processed_predictions = []
    # for pred in predictions:
    #     pred = pred[:, 0:1, :, :]
    #     processed_predictions.append(pred)

    sample = torch.cat((input.data, target, *predictions), 0)
    pred_img = predictions[-1]

    # sample_min = sample.min()
    # sample_max = sample.max()
    # sample = (sample - sample_min) / (sample_max - sample_min)
    # sample = sample.clamp(0, 1)

    # pred_min = pred_img.min()
    # pred_max = pred_img.max()
    # pred_img = (pred_img - pred_min) / (pred_max - pred_min)
    # pred_img = pred_img.clamp(0, 1)

    save_image(sample, pred_folder + ('%d.png' % i),
               nrow=7, normalize=True, pad_value=255)

    save_image(pred_img, pred_folder + ('prediction_%d.png' % i),
               nrow=7, normalize=True, pad_value=255)

    print(f'Processed batch {i+1}/{len(val_data_loader)}')
    print(f'Success rate: {np.array(success_rate).mean()}')


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
