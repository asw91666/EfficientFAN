import torch
import ref
import pdb

def get_pred(hmap):
    """Get predicted 2D pose from heat map"""
    num_batch = hmap.shape[0]
    num_joint = hmap.shape[1]
    h = hmap.shape[2]
    w = hmap.shape[3]
    hmap = hmap.reshape(num_batch, num_joint, h*w)
    idx = torch.argmax(hmap, dim=2)
    pred = torch.zeros(num_batch, num_joint, 2).to('cuda')
    for i in range(num_batch):
        for j in range(num_joint):
            pred[i, j, 0], pred[i, j, 1] = idx[i, j] % w, idx[i, j] // w
    return pred

def compute_error(output, target, weight):
    """Compute normalized mean error (NME)"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    res_in = ref.res_in
    res_out = output.shape[2]
    res_ratio = res_in / res_out
    pred = get_pred(output)
    pred = pred * res_ratio + res_ratio/2
    val = torch.sqrt(torch.mul(((pred - target) ** 2).sum(2), weight))
    error = val.sum()/(weight.sum().item()*res_in) * 100.0
    return error

def compute_error_direct(output, target, weight):
    """Compute normalized mean error (NME)"""
    num_batch = output.shape[0]
    num_joint = output.shape[1]
    res_in = ref.res_in
    val = torch.sqrt(torch.mul(((output - target) ** 2).sum(2), weight))
    error = val.sum()/(weight.sum().item()*res_in) * 100.0
    return error

def compute_error_direct_efficientFAN(output, target, norm_factor, weight):
    """Compute normalized mean error (NME)
        output : [B,Landmark,2]
        target : [B,Landmark,2]
        norm_factor : [B,L]
        weight : [B,Landmark]
    """
    # calc L2 norm
    # val = torch.sqrt(torch.mul(((output - target) ** 2).sum(2), weight)) # [B,L]
    # # broadcasting
    # norm_factor = norm_factor.unsqueeze(1).expand_as(val) # [B] --> [B,1] --> [B,L]
    # val = val / norm_factor # [B,L]

    val = torch.sqrt(torch.mul(((output - target) ** 2).sum(2), weight)) / norm_factor  # [B,L]
    error = val.sum()/(weight.sum().item()) * 100.0
    return error