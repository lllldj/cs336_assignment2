import torch
import torch.nn as nn
import os
import timeit
import torch.cuda.nvtx as nvtx

from tokenizers import ByteLevelBPETokenizer
from tokenizers import AddedToken

from cs336_basics.myModule import toy_Dataloader
from cs336_basics.myModule import toy_Transformer_lm
from cs336_basics.myOptimizer import toy_AdamW
from cs336_basics.myFunctional import toy_cross_entry, slow_generate, save_check_point , load_check_point, cosine_warm_up_lr, toy_grad_clip

vocab_path = "/root/workspace/cs336/assignment2/my_output/vocab.json"
merges_path = "/root/workspace/cs336/assignment2/my_output/merges.txt"
train_data_path = "/root/workspace/cs336/assignment1/data/TinyStoriesV2-GPT4-train.txt"
val_data_path = "/root/workspace/cs336/assignment1/data/TinyStoriesV2-GPT4-valid.txt"
weight_path = "/root/workspace/cs336/assignment2/my_output/weights/"

tok = ByteLevelBPETokenizer(vocab_path, merges_path)
tok.add_special_tokens([AddedToken("<|endoftext|>", special=True)])
tok.get_vocab_size()

device = "cuda:0"
vocab_size = tok.get_vocab_size()
max_context_len = 128
d_model = 768
d_ff = 3072
theta = 10000
num_layer = 12
num_heads = 12 

batch_size = 4
#train_round = 40000

train_data_loader = toy_Dataloader(train_data_path, tok, max_context_len,4*batch_size ,device)
val_data_loader = toy_Dataloader(val_data_path, tok, max_context_len, 4*batch_size, device)
model = toy_Transformer_lm(vocab_size,max_context_len,d_model,num_layer,num_heads,d_ff,theta,device).to(device)
opt = toy_AdamW(model.parameters())

warm_up_iter = 10
test_iter = 10
totol_iter = warm_up_iter + test_iter

data_0,target_0 = train_data_loader.get_batch(batch_size)
forward_times = []
total_times = []
with nvtx.range("warm up forward"):
    for iter in range(warm_up_iter):
        out = model(data_0)
        loss = toy_cross_entry(out,target_0)
        torch.cuda.synchronize()
for iter in range(test_iter):
    t0 = timeit.default_timer()
    with nvtx.range("forward"):
        out = model(data_0)
        loss = toy_cross_entry(out,target_0)
    torch.cuda.synchronize()
    t1 = timeit.default_timer()
    dt = t1 - t0
    forward_times.append(dt)
with nvtx.range("warm up"):
    for iter in range(warm_up_iter):
        out = model(data_0)
        loss = toy_cross_entry(out,target_0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
for iter in range(test_iter):
    t0 = timeit.default_timer()
    with nvtx.range("forward"):
        out = model(data_0)
        loss = toy_cross_entry(out,target_0)
    with nvtx.range("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with nvtx.range("opt step"):
        opt.step()
    torch.cuda.synchronize()
    t1 = timeit.default_timer()
    dt = t1 - t0
    total_times.append(dt)
print("forward time: {}, total time: {}".format(sum(forward_times)/test_iter, sum(total_times)/test_iter))
train_data_loader.close()
val_data_loader.close()

