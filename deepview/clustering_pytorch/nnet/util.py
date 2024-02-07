"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno
import scipy
import random

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' % (i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()




def gen_aug(sample, ssh_type):
    if ssh_type == 'na':
        return sample
    elif ssh_type == 'shuffle':  # 8
        return shuffle(sample)
    elif ssh_type == 'jit_scal':
        scale_sample = scaling(sample, sigma=2)
        return torch.from_numpy(scale_sample)
    elif ssh_type == 'perm_jit':
        return jitter(permutation(sample, max_segments=10), sigma=0.8)
    elif ssh_type == 'resample':
        return torch.from_numpy(resample(sample))
    elif ssh_type == 'noise':  # 1
        return jitter(sample)
    elif ssh_type == 'scale':  # 2
        return torch.from_numpy(scaling(sample))
    elif ssh_type == 'negate':  # 4
        return negated(sample)
    elif ssh_type == 't_flip':
        return time_flipped(sample)
    elif ssh_type == 'rotation':  # 3
        if isinstance(multi_rotation(sample), np.ndarray):
            return torch.from_numpy(multi_rotation(sample))
        else:
            return multi_rotation(sample)
    elif ssh_type == 'perm':  # 6
        return permutation(sample, max_segments=10)
    elif ssh_type == 't_warp':  # 7
        return torch.from_numpy(time_warp(sample))
    elif ssh_type == 'hfc':
        fft, fd = generate_high(sample, r=(32,2), high=True)
        return fd
    elif ssh_type == 'lfc':
        fft, fd = generate_high(sample, r=(32,2), high=False)
        return fd
    elif ssh_type == 'p_shift':
        return ifft_phase_shift(sample)
    elif ssh_type == 'ap_p':
        return ifft_amp_phase_pert(sample)
    elif ssh_type == 'ap_f':
        return ifft_amp_phase_pert_fully(sample)
    else:
        print('The task is not available!\n')



def shuffle(x):
    sample_ssh = []
    for data in x:
        p = np.random.RandomState(seed=21).permutation(data.shape[1])
        data = data[:, p]
        sample_ssh.append(data)
    return torch.stack(sample_ssh)


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1): # apply same distortion to the signals from each sensor
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[1]))
    ai = []
    for i in range(x.shape[2]):
        xi = x[:, :, i]
        ai.append(np.multiply(xi, factor[:, :])[:, :, np.newaxis])
    return np.concatenate((ai), axis=2)


def negated(X):
    return X * -1


def time_flipped(X):
    inv_idx = torch.arange(X.size(1) - 1, -1, -1).long()
    return X[:, inv_idx, :]


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp, :]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


def resample(x):
    from scipy.interpolate import interp1d
    orig_steps = np.arange(x.shape[1])
    interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/3)
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)
    start = random.choice(orig_steps)
    resample_index = np.arange(start, 3 * x.shape[1], 2)[:x.shape[1]]
    return InterpVal[:, resample_index, :]


def multi_rotation(x):
    n_channel = x.shape[2]
    n_rot = n_channel // 3
    x_rot = np.array([])
    for i in range(n_rot):
        x_rot = np.concatenate((x_rot, rotation(x[:, :, i * 3:i * 3 + 3])), axis=2) if x_rot.size else rotation(
            x[:, :, i * 3:i * 3 + 3])
    return x_rot

def rotation(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)
    return np.matmul(X, matrices)

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    """
    Get values for the cubic spline interpolation
    """
    cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def time_warp(X, sigma=0.2, num_knots=4):
    """
    Stretching and warping the time-series
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    X_transformed = np.empty(shape=X.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed


def distance(i, j, imageSize, r):
    dis_x = np.sqrt((i - imageSize[0] / 2) ** 2)
    dis_y =  np.sqrt((j - imageSize[1] / 2) ** 2)
    if dis_x < r[0] and dis_y < r[1]:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=(rows, cols), r=r)
    return mask


def generate_high(sample, r, high=True):
    # r: int, radius of the mask
    images = torch.unsqueeze(sample, 1)
    mask = mask_radial(torch.zeros([images.shape[2], images.shape[3]]), r)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1))) # shift: low f in the center
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    if high:
        fd = fd * (1.-mask)
    else:
        fd = fd * mask
    fft = torch.real(fd)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = torch.squeeze(fd.reshape([bs, c, h, w]))
    return fft, fd


def ifft_phase_shift(sample):
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp = fd.abs()
    phase = fd.angle()

    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, size=(sample.shape[0], sample.shape[1])), axis=2), sample.shape[2], axis=2)
    phase = phase + angles

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]))

    return ifft


def ifft_amp_phase_pert(sample):
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp = fd.abs()
    phase = fd.angle()

    # select a segment to conduct perturbations
    start = np.random.randint(0, int(0.5 * sample.shape[1]))
    end = start + int(0.5 * sample.shape[1])

    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, size=(sample.shape[0], sample.shape[1])), axis=2), sample.shape[2], axis=2)
    phase[:, start:end, :] = phase[:, start:end, :] + angles[:, start:end, :]

    # amp shift
    amp[:, start:end, :] = amp[:, start:end, :] + np.random.normal(loc=0., scale=0.8, size=sample.shape)[:, start:end, :]

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]))

    return ifft


def ifft_amp_phase_pert_fully(sample):
    images = torch.unsqueeze(sample, 1)
    bs, c, h, w = images.shape
    x = images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))

    amp = fd.abs()
    phase = fd.angle()

    # phase shift
    angles = np.repeat(np.expand_dims(np.random.uniform(low=-np.pi, high=np.pi, size=(sample.shape[0], sample.shape[1])), axis=2), sample.shape[2], axis=2)
    phase = phase + angles

    # amp shift
    amp = amp + np.random.normal(loc=0., scale=0.8, size=sample.shape)

    cmp = amp * torch.exp(1j * phase)
    ifft = torch.squeeze(torch.real(torch.fft.ifftn(torch.fft.ifftshift(cmp), dim=(-2, -1))).reshape([bs, c, h, w]))

    return ifft