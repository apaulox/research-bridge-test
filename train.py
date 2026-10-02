#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models.criterion import DenseAVContrastiveLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def create_optimizer(nets, opt):
    (net_visual, net_audio) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audio.parameters(), 'lr': opt.lr_audio}]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def save_training_state(path, next_epoch, total_steps, best_err,
                        net_visual, net_audio, optimizer, contrastive_criterion):
    state = {
        'next_epoch': next_epoch,
        'total_steps': total_steps,
        'best_err': best_err,
        'net_visual': net_visual.state_dict(),
        'net_audio': net_audio.state_dict(),
        'optimizer': optimizer.state_dict(),
        'contrastive_criterion': contrastive_criterion.state_dict(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path + '.tmp'
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)

def load_training_state(path, net_visual, net_audio, optimizer,
                        contrastive_criterion, device):
    state = torch.load(path, map_location=device, weights_only=False)
    net_visual.load_state_dict(state['net_visual'])
    net_audio.load_state_dict(state['net_audio'])
    optimizer.load_state_dict(state['optimizer'])
    contrastive_criterion.load_state_dict(state['contrastive_criterion'])

    random.setstate(state['python_random_state'])
    np.random.set_state(state['numpy_random_state'])
    torch.set_rng_state(state['torch_random_state'].cpu())
    if torch.cuda.is_available() and state.get('cuda_random_state') is not None:
        cuda_random_state = [rng_state.cpu() for rng_state in state['cuda_random_state']]
        torch.cuda.set_rng_state_all(cuda_random_state)

    return (
        int(state['next_epoch']),
        int(state['total_steps']),
        float(state['best_err']),
    )

#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                b = opt.batchSize
                loss = loss_criterion(output['binaural_spectrogram'][:b], output['audio_gt'][:b])
                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")
random.seed(opt.seed)
np.random.seed(opt.seed % (2**32))
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training clips = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
    #temperally set to val to load val data
    opt.mode = 'val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#validation clips = %d' % dataset_size_val)
    opt.mode = 'train' #set it back

if opt.tensorboard:
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# network builders
builder = ModelBuilder()
net_visual = builder.build_visual(
        weights=opt.weights_visual,
        backbone=opt.visual_backbone,
        dinov3_repo=opt.dinov3_repo,
        dinov3_weights=opt.dinov3_weights)
net_audio = builder.build_audio(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_audio)
nets = (net_visual, net_audio)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
loss_criterion = torch.nn.MSELoss()
contrastive_criterion = DenseAVContrastiveLoss()

if(len(opt.gpu_ids) > 0):
    loss_criterion.cuda(opt.gpu_ids[0])
    contrastive_criterion.cuda(opt.gpu_ids[0])

# initialization
start_epoch = 1
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_loss = []
batch_loss_mse = []
batch_loss_sem = []
batch_loss_spa = []
best_err = float("inf")

training_state_path = opt.resume_path or os.path.join(
    opt.checkpoints_dir, opt.name, 'training_latest.pth')
resumed = False
if opt.resume:
    if os.path.isfile(training_state_path):
        start_epoch, total_steps, best_err = load_training_state(
            training_state_path, net_visual, net_audio, optimizer,
            contrastive_criterion, opt.device)
        resumed = True
        print('resumed training state from %s' % training_state_path)
        print('continuing at epoch %d, total_steps %d, best validation %.6f' %
              (start_epoch, total_steps, best_err))
    else:
        print('no training state found at %s; starting a new run' % training_state_path)

if not resumed:
    # Model initialization consumes RNG; start data sampling from the chosen seed.
    random.seed(opt.seed)
    np.random.seed(opt.seed % (2**32))
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

for epoch in range(start_epoch, opt.niter+1):
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        for i, data in enumerate(dataset):
                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_loaded_time = time.time()

                total_steps += opt.batchSize

                # forward pass
                model.zero_grad()
                output = model.forward(data)

                b = opt.batchSize

                # compute mask mse loss
                loss_mse = loss_criterion(output['binaural_spectrogram'][:b], output['audio_gt'][:b])
                
                # compute contrastive loss
                s_samples = getattr(opt, 'spatial_num_samples', 0)
                contrastive_out = contrastive_criterion(
                    v_sem=output['semantic_visual_feat'][:b], 
                    a_sem=output['semantic_audio_feat'][:b],
                    v_spa=output['spatial_visual_feat'], 
                    a_spa=output['spatial_audio_feat'],
                    s=s_samples,
                    norm_semantic=getattr(opt, 'norm_semantic', False),
                    norm_spatial=getattr(opt, 'norm_spatial', False)
                )
                loss_contrastive = contrastive_out['loss']
                
                loss = loss_mse + loss_contrastive
                
                batch_loss.append(loss.item())
                batch_loss_mse.append(loss_mse.item())
                batch_loss_sem.append(contrastive_out['loss_semantic'].item())
                batch_loss_spa.append(contrastive_out['loss_spatial'].item())

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_forwarded_time = time.time()

                # update optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if(opt.measure_time):
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_steps // opt.batchSize % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        avg_loss = sum(batch_loss) / len(batch_loss)
                        avg_loss_mse = sum(batch_loss_mse) / len(batch_loss_mse)
                        avg_loss_sem = sum(batch_loss_sem) / len(batch_loss_sem)
                        avg_loss_spa = sum(batch_loss_spa) / len(batch_loss_spa)
                        print('Average loss: %.3f (mse: %.3f, sem: %.3f, spa: %.3f)' % (avg_loss, avg_loss_mse, avg_loss_sem, avg_loss_spa))
                        batch_loss = []
                        batch_loss_mse = []
                        batch_loss_sem = []
                        batch_loss_spa = []
                        if opt.tensorboard:
                            writer.add_scalar('data/loss', avg_loss, total_steps)
                            writer.add_scalar('data/loss_mse', avg_loss_mse, total_steps)
                            writer.add_scalar('data/loss_sem', avg_loss_sem, total_steps)
                            writer.add_scalar('data/loss_spa', avg_loss_spa, total_steps)
                        if(opt.measure_time):
                                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
                        torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_latest.pth'))

                if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
                        model.eval()
                        opt.mode = 'val'
                        print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        val_err = display_val(model, loss_criterion, writer, total_steps, dataset_val, opt)
                        print('end of display \n')
                        model.train()
                        opt.mode = 'train'
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                            torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                            torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_best.pth'))

                if(opt.measure_time):
                        iter_start_time = time.time()

        if(epoch % opt.save_epoch_freq == 0):
                print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_visual.pth'))
                torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_audio.pth'))

        #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
        if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
            decrease_learning_rate(optimizer, opt.decay_factor)
            print('decreased learning rate by ', opt.decay_factor)

        save_training_state(
            training_state_path, epoch + 1, total_steps, best_err,
            net_visual, net_audio, optimizer, contrastive_criterion)
        print('saved resumable training state for epoch %d to %s' %
              (epoch + 1, training_state_path))

if writer is not None:
    writer.close()
