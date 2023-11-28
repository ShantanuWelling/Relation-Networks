from __future__ import print_function

import json
import os
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils
import math
from clevr_dataset_connector import ClevrDatasetStateDescription
from model import RN


# Hyper parameters
lr = 0.000005
lr_max = 0.0005
lr_gamma = 2
lr_step = 20
seed = 42
batch_size = 640
dropout = -1
epochs = 200
bs_max = 1
bs_gamma = 1
bs_step = 20
clip_norm = 50


# Setting variables
print_freq = 10
clevr_dir = "../CLEVR_v1.0/"


def train(data, model, optimizer, epoch):
    model.train()

    avg_loss = 0.0
    n_batches = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, torch.cuda.is_available(), True)

        # forward and backward pass
        optimizer.zero_grad()
        output = model(img, qst)
        loss = F.nll_loss(output, label)
        loss.backward()

        # Gradient Clipping
        if clip_norm:
            clip_grad_norm(model.parameters(), clip_norm)

        optimizer.step()

        # Show progress
        progress_bar.set_postfix(dict(loss=loss.item()))
        avg_loss += loss.item()
        n_batches += 1

        if batch_idx % print_freq == 0:
            avg_loss /= n_batches
            processed = batch_idx * batch_size
            n_samples = len(data) * batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0


def test(data, model, epoch, dictionaries, test_results_dir):
    model.eval()

    # accuracy for every class
    class_corrects = {}
    # for every class, among all the wrong answers, how much are non pertinent
    class_invalids = {}
    # total number of samples for every class
    class_n_samples = {}
    # initialization
    for c in dictionaries[2].values():
        class_corrects[c] = 0
        class_invalids[c] = 0
        class_n_samples[c] = 0

    corrects = 0.0
    invalids = 0.0
    n_samples = 0

    inverted_answ_dict = {v: k for k,v in dictionaries[1].items()}
    sorted_classes = sorted(dictionaries[2].items(), key=lambda x: hash(x[1]) if x[1]!='number' else int(inverted_answ_dict[x[0]]))
    sorted_classes = [c[0]-1 for c in sorted_classes]

    confusion_matrix_target = []
    confusion_matrix_pred = []

    sorted_labels = sorted(dictionaries[1].items(), key=lambda x: x[1])
    sorted_labels = [c[0] for c in sorted_labels]
    sorted_labels = [sorted_labels[c] for c in sorted_classes]

    avg_loss = 0.0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, torch.cuda.is_available(), True, volatile=True)

        output = model(img, qst)
        pred = output.data.max(1)[1]

        loss = F.nll_loss(output, label)

        # compute per-class accuracy
        pred_class = [dictionaries[2][o.item()+1] for o in pred]
        real_class = [dictionaries[2][o.item()+1] for o in label.data]
        for idx,rc in enumerate(real_class):
            class_corrects[rc] += (pred[idx] == label[idx])
            class_n_samples[rc] += 1

        for pc, rc in zip(pred_class,real_class):
            class_invalids[rc] += (pc != rc)

        for p,l in zip(pred, label.data):
            confusion_matrix_target.append(sorted_classes.index(l))
            confusion_matrix_pred.append(sorted_classes.index(p))

        # compute global accuracy
        corrects += (pred == label).sum()
        assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
        invalids = sum(class_invalids.values())
        n_samples += len(label)
        assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'

        avg_loss += loss.item()

        if batch_idx % print_freq == 0:
            accuracy = corrects / n_samples
            invalids_perc = invalids / n_samples
            progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy), inv='{:.2%}'.format(invalids_perc)))

    avg_loss /= len(data)
    invalids_perc = invalids / n_samples
    accuracy = corrects / n_samples

    print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{}); Invalids = {:.2%} ({:g}/{}); Test loss = {}'.format(epoch, accuracy, corrects, n_samples, invalids_perc, invalids, n_samples, avg_loss))
    for v in class_n_samples.keys():
        accuracy = 0
        invalid = 0
        if class_n_samples[v] != 0:
            accuracy = class_corrects[v] / class_n_samples[v]
            invalid = class_invalids[v] / class_n_samples[v]
        print('{} -- acc: {:.2%} ({}/{}); invalid: {:.2%} ({}/{})'.format(v,accuracy,class_corrects[v],class_n_samples[v],invalid,class_invalids[v],class_n_samples[v]))

    # dump results on file
    filename = os.path.join(test_results_dir, 'test.pickle')
    dump_object = {
        'class_corrects':class_corrects,
        'class_invalids':class_invalids,
        'class_total_samples':class_n_samples,
        'confusion_matrix_target':confusion_matrix_target,
        'confusion_matrix_pred':confusion_matrix_pred,
        'confusion_matrix_labels':sorted_labels,
        'global_accuracy':accuracy
    }
    pickle.dump(dump_object, open(filename,'wb'))
    return avg_loss

def reload_loaders(clevr_dataset_train, clevr_dataset_test, train_bs, test_bs, state_description = False):

    clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs,
                                    shuffle=True, collate_fn=utils.collate_samples_state_description)
    clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, collate_fn=utils.collate_samples_state_description)
    return clevr_train_loader, clevr_test_loader

def initialize_dataset(clevr_dir, dictionaries, state_description=True):
    clevr_dataset_train = ClevrDatasetStateDescription(clevr_dir, True, dictionaries)
    clevr_dataset_test = ClevrDatasetStateDescription(clevr_dir, False, dictionaries)

    return clevr_dataset_train, clevr_dataset_test


#load hyperparameters from configuration file
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
with open('config.json') as config_file:
    hyp = json.load(config_file)['hyperparams']['original-sd']
#override configuration dropout
if dropout > 0:
    hyp['dropout'] = dropout

model_dirs = './model_{}_drop{}_bstart{}_bstep{}_bgamma{}_bmax{}_lrstart{}_'+ \
                    'lrstep{}_lrgamma{}_lrmax{}_invquests-{}_clipnorm{}_glayers{}_qinj{}_fc1{}_fc2{}'
model_dirs = model_dirs.format(
                    'original-sd', hyp['dropout'], batch_size, bs_step, bs_gamma,
                    bs_max, lr, lr_step, lr_gamma, lr_max,
                    True, clip_norm, hyp['g_layers'], hyp['question_injection_position'],
                    hyp['f_fc1'], hyp['f_fc2'])
if not os.path.exists(model_dirs):
    os.makedirs(model_dirs)
#create a file in this folder containing the overall configuration
hyp_str = str(hyp)
all_configuration = "all"+'\n\n'+hyp_str
filename = os.path.join(model_dirs,'config.txt')
with open(filename,'w') as config_file:
    config_file.write(all_configuration)

features_dirs = './features'
test_results_dir = './test_results'
if not os.path.exists(test_results_dir):
    os.makedirs(test_results_dir)


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

dictionaries = utils.build_dictionaries(clevr_dir)

clevr_dataset_train, clevr_dataset_test  = initialize_dataset(clevr_dir, dictionaries, hyp['state_description'])

# Build the model
qdict_size = len(dictionaries[0])
adict_size = len(dictionaries[1])

model = RN(hyp, qdict_size, adict_size)

if torch.cuda.device_count() > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    model.module.cuda()  # call cuda() overridden method

if torch.cuda.is_available():
    model.cuda()

start_epoch = 1

progress_bar = trange(start_epoch, epochs + 1)

bs = batch_size

# perform a full training
candidate_lr = lr * lr_gamma ** (start_epoch-1 // lr_step)
lr = candidate_lr if candidate_lr <= lr_max else lr_max

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-6, verbose=True)
scheduler = lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_gamma)
scheduler.last_epoch = start_epoch
for epoch in progress_bar:

    if(((bs_max > 0 and bs < bs_max) or bs_max < 0 ) and (epoch % bs_step == 0 or epoch == start_epoch)):
        bs = math.floor(batch_size * (bs_gamma ** (epoch // bs_step)))
        if bs > bs_max and bs_max > 0:
            bs = bs_max
        clevr_train_loader, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, bs, 640, hyp['state_description'])

        print('Dataset reinitialized with batch size {}'.format(bs))

    if((lr_max > 0 and scheduler.get_lr()[0]<lr_max) or lr_max < 0):
        scheduler.step()

    print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))

    # TRAIN
    progress_bar.set_description('TRAIN')
    train(clevr_train_loader, model, optimizer, epoch)

    # TEST
    progress_bar.set_description('TEST')
    test(clevr_test_loader, model, epoch, dictionaries, test_results_dir)

    # SAVE MODEL
    filename = 'RN_epoch_{:02d}.pth'.format(epoch)
    torch.save(model.state_dict(), os.path.join(model_dirs, filename))
