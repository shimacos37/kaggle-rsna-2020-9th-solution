import math
import torch

try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass
from torch.utils.data.distributed import DistributedSampler


class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = xm.xrt_world_size()
        if rank is None:
            rank = xm.get_ordinal()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        weights_dict = {1: 1, 0: 2}
        weights = (
            dataset["negative_exam_for_pe"].apply(lambda x: weights_dict[x]).values
        )
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        weights = self.weights[indices]
        assert len(indices) == self.num_samples
        return (
            indices[i]
            for i in torch.multinomial(weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_weighted_sampler(dataset):
    sampler = WeightedDistributedSampler(dataset=dataset)
    return sampler


def get_sampler(sampler_name, **params):
    # print("sampler name:", sampler_name)
    f = globals().get("get_" + sampler_name)
    return f(**params)
