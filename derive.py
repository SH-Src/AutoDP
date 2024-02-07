import os.path

import numpy as np
import torch
import random

from tqdm import tqdm

from utils.experiments import *
from model.surrogate import *


def evaluate(surrogate, pop, args, device):
    archs = [p[0] for p in pop]
    archs = torch.LongTensor(archs)
    ts = torch.LongTensor(np.stack([p[1] for p in pop], axis=0))
    archs = archs.to(device)
    ts = ts.to(device)

    gains = surrogate(archs, ts)
    gains = gains.masked_fill_(ts == 0, -np.inf)
    gains = gains.data.cpu().numpy()
    gains = np.max(gains, axis=0)
    gains = gains[: args.num_task]

    return sum(gains)


def search(surrogate, args, device):
    initial_pops = []
    for _ in range(10):
        pop = []
        for i in range(args.budget):
            a = random_arch(args.steps, args.num_op)
            ts = np.random.randint(0, high=2, size=args.num_task)
            pop.append([a, np.concatenate((ts, np.zeros(25 - args.num_task, dtype=int)))])
        initial_pops.append(pop)

    # best_gain = -np.inf
    for _ in tqdm(range(args.iter)):
        for i in range(len(initial_pops)):
            sample_id = random.randint(0, args.budget - 1)
            sample_point = initial_pops[i][sample_id]
            new_point = mutation(sample_point[0], sample_point[1], args)
            new_pop = list(initial_pops[i])
            new_pop.pop(sample_id)
            new_pop.append(new_point)
            assert len(initial_pops[i]) == len(new_pop) == args.budget
            initial_pops.append(new_pop)

        gains = []
        new_pops = []
        for p in initial_pops:
            gain = evaluate(surrogate, p, args, device)
            gains.append(gain)
        gains = torch.Tensor(gains)
        _, indices = torch.topk(gains, k=10)

        for id in indices:
            new_pops.append(initial_pops[id])
        initial_pops = new_pops

    best_gain = -np.inf
    best_p = None
    for p in initial_pops:
        gain = evaluate(surrogate, p, args, device)
        if gain > best_gain:
            best_gain = gain
            best_p = p


    return best_p, best_gain


def mutation(a1, t1, args):
    numop = len(a1)
    a = list(a1)
    t = list(t1)
    idx = random.randint(0, numop - 1)
    a[idx] = random.randint(0, args.num_op - 1)
    idx2 = random.randint(0, args.num_task - 1)
    if t[idx2] == 1:
        t[idx2] = 0
    elif t[idx2] == 0:
        t[idx2] = 1
    else:
        raise ValueError('wrong')

    return [a, t]


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='./saved_models/active-25')
parser.add_argument('--num_op', type=int, default=5, help='number of hidden units')
parser.add_argument('--steps', type=int, default=3, help='number of hidden units')
parser.add_argument('--num_task', type=int, default=25, help='number of hidden units')
parser.add_argument('--budget', type=int, default=10, help='number of hidden units')
parser.add_argument('--iter', type=int, default=2000, help='number of hidden units')
parser.add_argument('--dim', type=int, default=256, help='number of hidden units')
parser.add_argument('--sdim', type=int, default=64, help='number of hidden units')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--round', '-r', type=int, default=28)
parser.add_argument('--num_worker', type=int, default=0, help='number of hidden units')

parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--slr', type=float, default=0.0001, help='learning rate')

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


args = parser.parse_args()
print(args)

backbone_avp = np.array([0.5647, 0.4578, 0.1761, 0.5168, 0.4383, 0.2689, 0.4045, 0.1880, 0.5129, 0.5589, 0.5559, 0.3355,
                        0.5816, 0.5258, 0.6129, 0.1281, 0.4243, 0.2303, 0.1417, 0.2228, 0.1417, 0.3786, 0.5497, 0.4866, 0.5574])

backbone_roc = np.array([0.7827, 0.9079, 0.7226, 0.6948, 0.7296, 0.6791, 0.7229, 0.6712, 0.7601, 0.7351, 0.8844, 0.7484,
                        0.6730, 0.6298, 0.7396, 0.7076, 0.7141, 0.6849, 0.6371, 0.7602, 0.7051, 0.8171, 0.8651, 0.8291, 0.8792])

round = args.round
# surrogate = torch.load(os.path.join(args.save_dir2, 'model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
surrogate = torch.load(args.save_dir + '/model-{}.pt'.format(round))
surrogate.to(device)
pop, current_gain = search(surrogate, args, device)

print(pop)
print(current_gain)

torch.save(pop, os.path.join(args.save_dir, 'pop-{}-{}.pt'.format(round, args.budget)))

# pop = torch.load(os.path.join(args.save_dir, 'pop-{}.pt'.format(round)))

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

_, _, _, models = eval_samples(args, [p[0] for p in pop], [p[1] for p in pop], train_dl, val_dl, device)

pop = torch.load(os.path.join(args.save_dir, 'pop-{}-{}.pt'.format(round, args.budget)))
roc, f1, avp = eval_pop(args, [p[0] for p in pop], [p[1] for p in pop], test_dl, device)
# print(roc)
# print(f1)
avp = np.stack(avp, axis=0)
avp = np.max(avp, axis=0)
print(avp)

print(sum((avp[:args.num_task] - backbone_avp[:args.num_task])/backbone_avp[:args.num_task])/args.num_task)

roc = np.stack(roc, axis=0)
roc = np.max(roc, axis=0)
print(roc)
print(sum((roc[:args.num_task] - backbone_roc[:args.num_task])/backbone_roc[:args.num_task])/args.num_task)





