import torch
import torch.nn as nn

def total_loss(agent, actual_policy, old_policy, ADV_err, action, pi_1, k=3):
    kldiv_loss = nn.KLDivLoss(reduction="batchmean")

    beta = 0.5
    omega = 2.0
    omega_12 = 0.25
    
    value_loss_pg   = loss_pg(actual_policy[0], pi_1, ADV_err, action)
    value_loss_ppo  = loss_ppo(kldiv_loss, actual_policy, old_policy, beta, omega, k)
    value_loss_casc = loss_casc(kldiv_loss, actual_policy, old_policy, omega, omega_12, k)
    #loss = value_loss_pg + value_loss_ppo + value_loss_casc
    #print(f"Total loss value: {loss}\n")

    loss = value_loss_ppo
    return loss


def loss_pg(actual_policy, pi_1, ADV_err, action):
    loss_tensor = (actual_policy[action] / pi_1[action]) * ADV_err
    loss_tensor = loss_tensor.clone().detach()

    loss = loss_tensor.clone().detach().requires_grad_(True)
    #print(f"PG loss value: {loss}")
    return loss

def loss_ppo(kldiv_loss, actual_policy, old_policy, beta, omega, k=3):
    # k -----> set of policies used in Synaptic Consolidation
    loss = 0
    beta = 8.0
    omega = 4.0

    for i in range(0, k):
        loss += -beta*(omega**(i))*kldiv_loss(old_policy[i], actual_policy[i]) 
    # KLDiv(output of model, observated experience) ---> the output of the model must be computed with logSoftmax, not Softmax
    
    #print(f"PPO loss value: {loss}, kldiv: {kldiv_loss(old_policy[i], actual_policy[i]) }\nactual_policy: {actual_policy}, old_policy: {old_policy}\n")
    return loss

def loss_casc(kldiv_loss, actual_policy, old_policy, omega, omega_12, k=3):
    loss = -(omega_12)*kldiv_loss(old_policy[1], actual_policy[0]) 
    for i in range(1, k-1):
        loss += -(omega)*kldiv_loss(old_policy[i-1], actual_policy[i]) - kldiv_loss(old_policy[i+1], actual_policy[i])
    loss += -(omega)*kldiv_loss(old_policy[k-2], actual_policy[k-1]) - kldiv_loss(old_policy[k-1], actual_policy[k-1])
    
    #print(f"CASC loss value: {loss}")
    return loss