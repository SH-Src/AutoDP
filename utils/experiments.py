import os
import random

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from model.model_ import *
from utils.preprocessing import *
from utils.ehr_dataset import *
from model.surrogate import *
import torch
import torch.nn as nn

def eval_metric(eval_set, model, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_score = []

        for i, data in tqdm(enumerate(eval_set)):
            x, y, length = data
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            x = x.to(device)
            y = y.to(device)
            length = torch.from_numpy(np.array(length)).long()
            length = length.to(device)
            logits = model(x, length)
            scores = logits
            scores = scores.data.cpu().numpy()
            labels = y.data.cpu().numpy()
            y_true.append(labels)
            y_score.append(scores)
        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)
        average_precision = average_precision_score(y_true, y_score, average=None)
        average_precision_macro = average_precision_score(y_true, y_score)
    return average_precision_macro, average_precision

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/phenotyping/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


def evaluate(args, archs, task_combs, train_dl, val_dl, device):
    models = nn.ModuleList()
    paths = []
    task_combs_tensor = []
    for arch, task_comb in zip(archs, task_combs):
        path = os.path.join(args.save_dir,
                            str(arch).strip('[').strip(']') + '-' + str(list(task_comb)).strip('[').strip(']'))
        if not os.path.exists(path):
            os.mkdir(path)
        model = Network(d_model=args.dim, steps=args.steps, genotype=arch)
        model.to(device)
        models += [model]
        task_combs_tensor.append(torch.LongTensor(task_comb).to(device))

        paths.append(path)

    loss_func = nn.BCELoss(reduction='sum')
    optim = torch.optim.Adam(models.parameters(), args.lr)
    best_results = np.zeros(len(models))
    predictions = [np.zeros(len(task_combs[0])) for _ in range(len(models))]
    trained_models = [None for _ in range(len(models))]
    global_step = 0
    for epoch_id in range(args.epochs):
        print('epoch: {:5} '.format(epoch_id))
        models.train()
        start_time = time.time()
        for i, data in enumerate(train_dl):
            x, y, length = data
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            x = x.to(device)
            y = y.to(device)
            length = torch.from_numpy(np.array(length)).long()
            length = length.to(device)
            optim.zero_grad()
            for model, taskcomb in zip(models, task_combs_tensor):
                out = model(x, length)
                loss = loss_func(out * taskcomb, y * taskcomb) / torch.sum(taskcomb)
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            x.cpu()
            y.cpu()
            length.cpu()
            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} | ms/batch {:7.2f} |'.format(global_step, ms_per_batch))
                total_loss = 0.0
                start_time = time.time()
            global_step += 1

        models.eval()
        for j in range(len(models)):
            model_path = os.path.join(paths[j] + '/model.pt')
            result_path = os.path.join(paths[j] + '/results.txt')
            gains_path = os.path.join(paths[j] + '/gains.npy')
            _, d_avgs = eval_metric(val_dl, models[j], device)
            avg = sum(d_avgs * np.array(task_combs[j])) / sum(np.array(task_combs[j]))
            if avg >= best_results[j]:
                best_results[j] = avg
                predictions[j] = d_avgs * np.array(task_combs[j])
                trained_models[j] = models[j]
                torch.save(models[j], model_path)
                with open(result_path, 'w') as fout:
                    for a in archs[j]:
                        fout.write(str(a))
                    fout.write('\n')
                    for t in task_combs[j]:
                        fout.write(str(int(t)))
                    fout.write('\n')
                    fout.write(str(d_avgs * np.array(task_combs[j])))
                np.save(gains_path, d_avgs * np.array(task_combs[j]))
            torch.save(epoch_id, os.path.join(paths[j] + '/epoch.pt'))
    return predictions, trained_models

def eval_samples(args, archs, task_combs, train_dl, val_dl, device):
    to_train_tasks = []
    to_train_archs = []
    to_train_gains = []
    to_train_models = []
    _tasks = []
    _archs = []
    _gains = []
    _models = []
    for a, t in zip(archs, task_combs):
        path = os.path.join(args.save_dir,
                            str(a).strip('[').strip(']') + '-' + str(list(t)).strip('[').strip(']'))
        if os.path.exists(os.path.join(path + '/gains.npy')):
            epoch = torch.load(os.path.join(path + '/epoch.pt'))
            if epoch >= 12:
                gain = np.load(os.path.join(path + '/gains.npy'))
                model = torch.load(os.path.join(path + '/model.pt'))
                _tasks.append(t)
                _archs.append(a)
                _gains.append(gain)
                _models.append(model)
            else:
                to_train_archs.append(a)
                to_train_tasks.append(t)
        else:
            to_train_archs.append(a)
            to_train_tasks.append(t)
    if len(to_train_tasks) > 0:
        if len(to_train_tasks)<=10:
            to_train_gains, to_train_models = evaluate(args, to_train_archs, to_train_tasks, train_dl, val_dl, device)
        elif len(to_train_tasks)<=20:
            to_train_gains, to_train_models = evaluate(args, to_train_archs[0:10], to_train_tasks[0:10], train_dl, val_dl, device)
            to_train_gains1, to_train_models1 = evaluate(args, to_train_archs[10:], to_train_tasks[10:], train_dl, val_dl, device)
            to_train_gains += to_train_gains1
            to_train_models += to_train_models1
        else:
            to_train_gains, to_train_models = evaluate(args, to_train_archs[0:10], to_train_tasks[0:10], train_dl,
                                                       val_dl, device)
            to_train_gains1, to_train_models1 = evaluate(args, to_train_archs[10:20], to_train_tasks[10:20], train_dl,
                                                         val_dl, device)
            to_train_gains2, to_train_models2 = evaluate(args, to_train_archs[20:], to_train_tasks[20:], train_dl,
                                                         val_dl, device)
            to_train_gains += to_train_gains1 + to_train_gains2
            to_train_models += to_train_models1 + to_train_models2
    _archs = _archs + to_train_archs
    _tasks = _tasks + to_train_tasks
    _gains = _gains + to_train_gains
    _models = _models + to_train_models
    return _archs, _tasks, _gains, _models

def eval_pop(args, archs, task_combs, test_dl, device):
    roc = []
    f1 = []
    avp = []
    for a, t in zip(archs, task_combs):
        path = os.path.join(args.save_dir,
                            str(a).strip('[').strip(']') + '-' + str(list(t)).strip('[').strip(']'))
        model = torch.load(os.path.join(path + '/model.pt'))
        model.eval()
        with torch.no_grad():
            y_true = []
            y_score = []
            y_pred = []
            for i, data in enumerate(test_dl):
                x, y, length = data
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
                x = x.to(device)
                y = y.to(device)
                length = torch.from_numpy(np.array(length)).long()
                length = length.to(device)
                logits = model(x, length)
                scores = logits
                scores = scores.data.cpu().numpy()
                labels = y.data.cpu().numpy()
                pred = np.where(scores > 0.5, 1.0, 0.0)
                y_true.append(labels)
                y_score.append(scores)
                y_pred.append(pred)
            y_true = np.concatenate(y_true, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            average_precision = average_precision_score(y_true, y_score, average=None)
            f1score = f1_score(y_true, y_pred, average=None)
            rocauc = roc_auc_score(y_true, y_score, average=None)
            roc.append(rocauc*t)
            f1.append(f1score*t)
            avp.append(average_precision*t)
    return roc, f1, avp

def meta_train(model, args, archs, task_combs, gains, device):
    meta_data = MetaData(archs, task_combs, gains)
    dataloader = DataLoader(meta_data, batch_size=args.meta_bs, shuffle=True, collate_fn=meta_collate, pin_memory=True)
    loss_func = nn.L1Loss(reduction='sum')
    optim = torch.optim.Adam(model.parameters(), args.slr)
    model.to(device)
    losses = []
    for epoch in range(args.meta_eps):
        print('epoch: {:5} '.format(epoch))
        model.train()
        for i, data in enumerate(dataloader):
            arch, task, gain = data
            arch = arch.to(device)
            task = task.to(device)
            gain = gain.to(device)
            optim.zero_grad()
            loss = loss_func(model(arch, task) * task, gain * task) / torch.sum(task)
            loss.backward()
            optim.step()

        model.eval()
        total_loss = []
        for i, data in enumerate(dataloader):
            arch, task, gain = data
            arch = arch.to(device)
            task = task.to(device)
            gain = gain.to(device)
            loss = loss_func(model(arch, task) * task, gain * task) / torch.sum(task)
            total_loss.append(loss.item())
        total_loss = np.mean(total_loss)
        print(total_loss)
        losses.append(total_loss)
        if total_loss < 0.005:
            break
    return model, losses

def random_arch(steps, operations):
    result = []
    for i in range(steps):
        for j in range(i+1):
            result.append(random.randint(0, operations-1))

    return result

