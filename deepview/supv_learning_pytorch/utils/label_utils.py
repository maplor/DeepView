import numpy as np
import torch


# https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321
def to_one_hot(y, n_classes):
    '''
    y: shape(BS, T, 1)
    y_onehot: shape(BS, T, 1, n_classes)
    '''
    y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes)
    return y_onehot


def mixup_process(x, y, lam):
    '''
    x: shape(BS, CH, T, 1)
    y (y_onehot): shape(BS, T, 1, n_classes)
    lam: shape()
    '''
    # print(f"mixup_alpha = {mixup_alpha}")
    batch_size = x.size()[0]
    y = y.to(torch.float32)
    indices = np.random.permutation(batch_size)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # expand lambda
    # log.debug(f"x.shape: {x.shape}")
    # log.debug(f"x.dim(): {x.dim()}")
    # log.debug(f"y.shape: {y.shape}")
    # log.debug(f"y.dim(): {y.dim()}")
    x_last_dim_idx = int(x.dim() - 1)
    y_last_dim_idx = int(y.dim() - 1)
    lam_x = lam.expand_as(x.transpose(0, x_last_dim_idx))
    lam_x = lam_x.transpose(0, x_last_dim_idx)
    lam_y = lam.expand_as(y.transpose(0, y_last_dim_idx))
    lam_y = lam_y.transpose(0, y_last_dim_idx)
    # lam_x = lam.expand_as(x.transpose(0, 3))
    # lam_x = lam_x.transpose(0, 3)
    # lam_y = lam.expand_as(y.transpose(0, 3))
    # lam_y = lam_y.transpose(0, 3)

    # mixup
    x_mixed = x * lam_x + x_shuffled * (1 - lam_x)
    y_mixed = y * lam_y + y_shuffled * (1 - lam_y)

    return x_mixed, y_mixed


def get_label_species(cfg):
    label_species = cfg.dataset.species
    return label_species


def convert_torch_labels(targets, label_species):
    if label_species == "om":
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 1, targets)  # bathing
        targets = torch.where(targets == 400, 2, targets)  # flight_take_off
        targets = torch.where(targets == 401, 3, targets)  # flight_cruising
        targets = torch.where(targets == 501, 4, targets)  # foraging_dive
        targets = torch.where(targets == 502, 5, targets)  # surface_seizing (dipping)
    elif label_species == "um":
        targets = torch.where(targets == 100, 0, targets)  # ground_stationary
        targets = torch.where(targets == 101, 1, targets)  # ground_active
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 2, targets)  # bathing
        targets = torch.where(targets == 301, 2, targets)  # bathing
        targets = torch.where(targets == 400, 3, targets)  # flying_active
        targets = torch.where(targets == 401, 4, targets)  # flying_passive
        targets = torch.where(targets == 500, 5, targets)  # foraging
        targets = torch.where(targets == 501, 5, targets)  # foraging_poss
        targets = torch.where(targets == 502, 5, targets)  # foraging_fish
        targets = torch.where(targets == 503, 5, targets)  # foraging_fish_poss
        # targets = torch.where(targets==510, 6, targets) # foraging_insect
        # targets = torch.where(targets==511, 6, targets) # foraging_insect_poss
        # targets = torch.where(targets==520, 7, targets) # foraging_something

    return targets


def return_species_jp_name(cfg):
    if cfg.dataset.species == "om":
        species_jp_name = "omizunagidori"
    elif cfg.dataset.species == "um":
        species_jp_name = "umineko"
    else:
        raise Exception(f"cfg.dataset: {cfg.dataset} is unknonw dataset.")

    return species_jp_name


def generate_class_labels_for_vis(species):
    if species == "omizunagidori":
        class_label = [
            'Stationary',
            'Bathing',
            'Take-off',
            'Cruising Flight',
            'Foraging Dive',
            'Dipping'
        ]
    elif species == "umineko":
        class_label = [
            'Stationary',
            'Ground Active',
            'Bathing',
            'Active Flight',
            'Passive Flight',
            'Foraging'
        ]
    return class_label