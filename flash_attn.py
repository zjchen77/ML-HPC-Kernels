import torch
import torch.nn as nn
import numpy as np
import sys
import time
from einops import rearrange

BLOCK_SIZE = 1024
NEG_INF = -1e10 # -infinity
EPSILON = 1e-10

def normal_attention(Q, K, V, mask=None):
    scale = 1 / np.sqrt(Q.shape[-1])
    Q = Q * scale
    QKt = torch.einsum('... i d, ... j d -> ... i j', Q, K)

    key_mask = rearrange(mask, 'b j -> b 1 1 j')
    QKt = torch.where(key_mask > 0, QKt, NEG_INF)

    attn = nn.functional.softmax(QKt, dim=-1)
    return attn @ V

def flash_attention_forward(Q, K, V, mask=None):
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[...,None]    #削去在最后一维即dim维
    m = torch.ones(Q.shape[:-1])[...,None] * NEG_INF

    O = O.to(device='cuda')
    l = l.to(device='cuda')
    m = m.to(device='cuda')

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)

    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        maskj = mask_BLOCKS[j]

        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled  = Qi * scale

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
            
            # Masking
            maskj_temp = rearrange(maskj, 'b j -> b 1 1 j') # ?????
            S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            # Masking
            P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

            mi_new = torch.maximum(m_block_ij, mi)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij    #li_new 就是 Lall
            
            O_BLOCKS[i] = (li/li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new
        
    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)
    return O, l, m


def flash_attention(Q, K, V, mask):
    out = flash_attention_forward(Q, K, V, mask)
    return out[0]