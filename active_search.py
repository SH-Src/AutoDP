import os.path
import random

import numpy as np
import torch

from utils.experiments import *
import itertools

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--dim', type=int, default=256, help='number of hidden units')
parser.add_argument('--sdim', type=int, default=64, help='number of hidden units')
parser.add_argument('--num_op', type=int, default=5, help='number of hidden units')
parser.add_argument('--steps', type=int, default=3, help='number of hidden units')
parser.add_argument('--warm_archs', type=int, default=20, help='number of hidden units')
parser.add_argument('--num_task', type=int, default=25, help='number of hidden units')
parser.add_argument('--K', type=int, default=30, help='number of hidden units')
parser.add_argument('--p1', type=int, default=50, help='number of hidden units')
parser.add_argument('--p2', type=int, default=5, help='number of hidden units')
parser.add_argument('--warmup', type=int, default=-1, help='number of hidden units')

parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--meta_eps', type=int, default=1000)
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--meta_bs', '-mbs', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--num_classes', type=int, default=25)
parser.add_argument('--patience', type=int, default=15, help='number of epoch to wait for best')

parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--slr', type=float, default=0.00005, help='learning rate')

parser.add_argument('--max_grad_norm', type=float, default=5.0, help='clip gradients')
parser.add_argument('--log_interval', type=int, default=60, help='log')
parser.add_argument('--align', type=float, default=0.0, help='align weight')

parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
parser.add_argument('--normalizer_state', type=str, default='pheno_ts.1.00_impute.previous_start.zero_masks.True_n.48329.normalizer',
                        help='Path to a state file of a normalizer. Leave none if you want to '
                                'use one of the provided ones.')

parser.add_argument('--ehr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
                        default='data')

parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='evaluations-25')

parser.add_argument('--save_dir2', type=str, help='Directory relative which all output files are stored',
                    default='saved_models/active-25')


args = parser.parse_args()
print(args)

discretizer = Discretizer(timestep=float(1.0),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
normalizer.load_params(normalizer_state)

train_dl, val_dl, test_dl = get_data_loader(discretizer, normalizer, args, args.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


backbone_score = np.array([0.5647, 0.4578, 0.1761, 0.5168, 0.4383, 0.2689, 0.4045, 0.1880, 0.5129, 0.5589, 0.5559, 0.3355,
                        0.5816, 0.5258, 0.6129, 0.1281, 0.4243, 0.2303, 0.1417, 0.2228, 0.1417, 0.3786, 0.5497, 0.4866, 0.5574])

#warm up
if args.warmup == -1:
    surrogate = Surrogate(args.sdim, args.num_op, args.steps)
    collected_a, collected_t, collected_g = [], [], []
    print('parameters:')
    for name, param in surrogate.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in surrogate.parameters() if p.requires_grad)

    print('\ttotal:', num_params)

    surrogate.to(device)
    task_comb = np.concatenate((np.ones(args.num_task, dtype=int), np.zeros(25 - args.num_task, dtype=int)))
    archs = [random_arch(args.steps, args.num_op) for _ in range(args.warm_archs)]
    a, t, g, m = eval_samples(args, archs, [task_comb] * len(archs), train_dl, val_dl, device)
    collected_a += a
    collected_t += t
    new_g = []
    for ta, ga in zip(t, g):
        gains = np.zeros(25)
        for j in range(len(ga)):
            if backbone_score[j] > 0.0:
                gains[j] = ga[j] * ta[j] / backbone_score[j]
        new_g.append(gains)
    collected_g += new_g

    surrogate, losses = meta_train(surrogate, args, collected_a, collected_t, collected_g, device)
    torch.save(surrogate, os.path.join(args.save_dir2, 'model-0.pt'))
    torch.save((collected_a, collected_t, collected_g), os.path.join(args.save_dir2, 'data-0.pt'))

else:
    surrogate = torch.load(os.path.join(args.save_dir2, 'model-{}.pt'.format(args.warmup)))
    collected_a, collected_t, collected_g = torch.load(os.path.join(args.save_dir2, 'data-{}.pt'.format(args.warmup)))

#active search

def forward_sur(surrogate, a, t, device):
    a = torch.LongTensor(a).to(device)
    t = torch.LongTensor(np.stack(t, axis=0)).to(device)
    gains = surrogate(a, t)
    return gains

for i in range(1, args.K + 1):
    if args.warmup == -1:
        round = i
    else:
        round = args.warmup + i
    print('Round: ', round)
    a_list, t_list, g_list = [], [], []
    for j in range(args.num_task):
        ts_set = [np.zeros(25, dtype=int) for _ in range(min(2**(args.num_task-1), 100))]
        tasks = [a for a in range(args.num_task) if not a == j]
        task_combs = []
        for t in range(args.num_task):
            for com in itertools.combinations(tasks, t):
                task_combs.append(com)
        if len(task_combs) > len(ts_set):
            task_combs = random.sample(task_combs, len(ts_set))
        for c in range(len(task_combs)):
            ts_set[c][j] = 1
            for t in task_combs[c]:
                ts_set[c][t] = 1

        archs_set = [[random_arch(args.steps, args.num_op) for _ in range(args.p1)] for _ in range(len(ts_set))]
        predicted_gains = []
        for a in range(len(archs_set)):
            g = forward_sur(surrogate, archs_set[a], [ts_set[a]]*args.p1, device)
            predicted_gains.append(g[:, j])

        gains_topk_mean = []
        indices_topk = []
        for g in predicted_gains:
            value, indices = torch.topk(g, k=args.p2)
            gains_topk_mean.append(torch.mean(value))
            indices_topk.append(indices)

        gains_topk_mean = torch.stack(gains_topk_mean)
        prob = torch.softmax(gains_topk_mean, dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=prob)
        sampled_task = dist.sample().item()
        idx = random.sample(list(indices_topk[sampled_task]), k=1)[0].item()
        sampled_arch = archs_set[sampled_task][idx]
        a_list.append(sampled_arch)
        t_list.append(ts_set[sampled_task])

    a, t, g, m = eval_samples(args, a_list, t_list, train_dl, val_dl, device)
    collected_a += a
    collected_t += t
    new_g = []
    for ta, ga in zip(t, g):
        gains = np.zeros(25)
        for j in range(len(ga)):
            if backbone_score[j] > 0.0:
                gains[j] = ga[j] * ta[j] / backbone_score[j]
        new_g.append(gains)
    collected_g += new_g
    surrogate, losses = meta_train(surrogate, args, collected_a, collected_t, collected_g, device)
    torch.save(surrogate, os.path.join(args.save_dir2, 'model-{}.pt'.format(round)))
    torch.save((collected_a, collected_t, collected_g), os.path.join(args.save_dir2, 'data-{}.pt'.format(round)))
    torch.save(losses, os.path.join(args.save_dir2, 'loss-{}.pt'.format(round)))


