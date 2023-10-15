import math
import torch

def _get_interleave(n):
    '''
    生成每一个对应head的权重
    '''
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

print(_get_interleave(8))

# [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

# 对应2^-1到2^-8

def _gen_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head)) # n_head
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1) 
    # slopes.unsqueeze(1).unsqueeze(1):[n_head, 1, 1]
    # torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1): [n_head, 1, max_pos]
    # 两者逐元素相乘,alibi:[n_head, 1, max_pos]
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    # _fill_with_neg_inf(torch.zeros([max_pos, max_pos]))：首先，创建一个形状为[max_pos, max_pos]的零矩阵，然后使用_fill_with_neg_inf函数将矩阵中的所有元素填充为负无穷。结果矩阵维度依然为[max_pos, max_pos]。
    # torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1): 使用torch.triu函数生成上三角矩阵，将传入矩阵的下半部分（对角线以下）置为0，只保留上半部分（对角线以上）的负无穷值。1参数表示从主对角线的下一行开始置为0。结果矩阵维度依然为[max_pos, max_pos]。
    # 总结起来，alibi_mask是一个形状为[max_pos, max_pos]的上三角矩阵，对角线以上的元素（不含对角线）为负无穷，其他位置为0。
    # alibi_mask起到将上三角矩阵mask的作用
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    # alibi_mask.unsqueeze(0) + alibi：将alibi_mask张量与alibi张量相加。由于它们的维度分别为(1, max_pos, max_pos)和(n_head, 1, max_pos)，所以这里的加法实际上是一种广播（broadcasting）操作。广播规则要求从最后一个维度开始逐个维度比较，如果有一个维度大小不同且其中一个为1，则将该维度扩展成较大的那个。在这个例子中，根据广播规则，alibi_mask将在第一个维度上扩展n_head次，alibi将在第二个维度上扩展max_pos次。因此，alibi_mask.unsqueeze(0) + alibi的结果张量维度为(n_head, max_pos, max_pos)。
    return alibi_mask

print(torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1))
# tensor([[[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]],

#         [[0, 1, 2, 3]]])

print(_gen_alibi_mask(8, 4))
# # tensor([[[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.5000,   -inf,   -inf],
#          [0.0000, 0.5000, 1.0000,   -inf],
#          [0.0000, 0.5000, 1.0000, 1.5000]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.2500,   -inf,   -inf],
#          [0.0000, 0.2500, 0.5000,   -inf],
#          [0.0000, 0.2500, 0.5000, 0.7500]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.1250,   -inf,   -inf],
#          [0.0000, 0.1250, 0.2500,   -inf],
#          [0.0000, 0.1250, 0.2500, 0.3750]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.0625,   -inf,   -inf],
#          [0.0000, 0.0625, 0.1250,   -inf],
#          [0.0000, 0.0625, 0.1250, 0.1875]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.0312,   -inf,   -inf],
#          [0.0000, 0.0312, 0.0625,   -inf],
#          [0.0000, 0.0312, 0.0625, 0.0938]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.0156,   -inf,   -inf],
#          [0.0000, 0.0156, 0.0312,   -inf],
#          [0.0000, 0.0156, 0.0312, 0.0469]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.0078,   -inf,   -inf],
#          [0.0000, 0.0078, 0.0156,   -inf],
#          [0.0000, 0.0078, 0.0156, 0.0234]],

#         [[0.0000,   -inf,   -inf,   -inf],
#          [0.0000, 0.0039,   -inf,   -inf],
#          [0.0000, 0.0039, 0.0078,   -inf],
#          [0.0000, 0.0039, 0.0078, 0.0117]]])