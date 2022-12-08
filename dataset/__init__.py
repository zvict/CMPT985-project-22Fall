from .dataset import RINDataset
from torch.utils.data import DataLoader


def get_traindataset(args):
    return RINDataset(args)


def get_trainloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


def get_testdataset(args):
    return RINDataset(args)


def get_testloader(dataset, args):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
