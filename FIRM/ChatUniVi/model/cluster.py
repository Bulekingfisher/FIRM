import torch
import math
import torch.nn as nn
import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    try:
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    except:
        return tensor

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def adaptive_cluster_centers(score, cluster_num_origin, r=None):
    B, N = score.shape
    if cluster_num_origin > N:
        cluster_num_origin = N
    score_flat = score.squeeze(0)
    num_max = min(int(cluster_num_origin*1.2),cluster_num_origin)
    num_min = int(cluster_num_origin*0.6)

    sorted_score, sorted_indices = torch.sort(score_flat, descending=True) 

    diff = torch.diff(sorted_score)

    top_diff_values, top_diff_indices = torch.topk(diff, k=num_max, largest=True, sorted=True)

    cluster_num = 0
    for i in range(len(top_diff_indices)):
        candidate_num = top_diff_indices[i].item() + 1
        if candidate_num >= num_min and candidate_num <= num_max:
            cluster_num = max(candidate_num,cluster_num)
    if cluster_num == 0:
        cluster_num = cluster_num_origin
    index_down = sorted_indices[:cluster_num].unsqueeze(0) 
    
    return index_down, cluster_num

def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None, r=None):
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape
        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

        if B == 1 and r: 
            t = torch.arange(N, device=x.device).float() 
            t_normalized = t / (N - 1) if N > 1 else t 
            
            tau_base = 0.1
            current_level = 1
            total_levels = 3
            alpha = 1.5 
            delta_t = torch.abs(t_normalized[:, None] - t_normalized[None, :])

            tau = tau_base * (1 + current_level / total_levels) ** 2 
            time_penalty = delta_t * torch.exp(tau * delta_t)/ C

            dist_matrix += time_penalty.unsqueeze(0)
            delta_idx = torch.abs(torch.arange(N, device=x.device)[:, None] - torch.arange(N, device=x.device)[None, :])
            dist_matrix[0][delta_idx > r] = 1e9

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                        (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False) 
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density

        if B == 1 and r:
            index_down, cluster_num = adaptive_cluster_centers(score, cluster_num, r) 
        else:
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num) 
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)

        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num
    
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight

    x_merged.index_add_(dim=0, index=idx.reshape(B * N), 
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = None
    return out_dict

class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.k = k

    def forward(self, token_dict, sample_ratio=None):
        x = token_dict["x"]
        B, N, C = x.shape

        token_weight = x.new_ones(B, N)

        if token_dict["mask"] is not None:
            token_weight.masked_fill_((1 - token_dict["mask"]).to(torch.bool), float("-inf"))
        token_weight = token_weight.unsqueeze(2)
        token_dict['x'] = x 

        if sample_ratio is not None:
            cluster_num = max(math.ceil(N * sample_ratio), 1)
        if self.sample_ratio > 1:
            cluster_num = max(math.ceil(self.sample_ratio), 1)
        else:
            cluster_num = max(math.ceil(N * self.sample_ratio), 1)

        k = min(3, max(cluster_num//2, 1)) if self.k > cluster_num else self.k
        if "r" in token_dict:
            idx_cluster, cluster_num = cluster_dpc_knn(
                token_dict, cluster_num, k, token_mask=token_dict["mask"], r=token_dict["r"])
        else:
            idx_cluster, cluster_num = cluster_dpc_knn(
                token_dict, cluster_num, k, token_mask=token_dict["mask"])

        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        down_dict['assign_from_prev'] = idx_cluster 
        return down_dict, token_dict

class TCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=False):
        super().__init__()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x']
        return q_dict