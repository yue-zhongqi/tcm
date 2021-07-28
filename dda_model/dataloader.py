from torch.utils.data import DataLoader
from cyclegan.base_dataset import get_transform
from data_list import ImageList, ExpertImageList


class DDALoader():
    def __init__(self, opt, dset_file, batch_size, noflip=False, return_path=False, drop_last=True, transform=None,
                 return_cdm=False, cdm_path="", cdm_transform=None, dset=None, center=False):
        if return_cdm:
            assert cdm_path != ""
        self.opt = opt
        if transform is None and dset is None:
            transform = get_transform(self.opt, grayscale=False, noflip=noflip, center=center)
        if dset is None:
            if opt.all_experts and return_cdm:
                self.dset = ExpertImageList(open(dset_file).readlines(), transform=transform, return_path=return_path,
                                            n_experts=opt.n_experts, cdm_path=cdm_path, cdm_transform=cdm_transform)
            else:
                self.dset = ImageList(open(dset_file).readlines(), transform=transform, return_path=return_path,
                                      return_cdm=return_cdm, cdm_path=cdm_path, cdm_transform=cdm_transform)
        else:
            self.dset = dset
        if opt.debug:
            num_workers = 0
        else:
            num_workers = 4
        self.loader = DataLoader(self.dset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=drop_last)
        self.length = len(self.loader)
        self.idx = 0

    def next(self):
        if self.idx % self.length == 0:
            self.iterator = iter(self.loader)
        self.idx += 1
        return self.iterator.next()