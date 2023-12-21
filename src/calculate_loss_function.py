import torch
import torch.nn as nn

def total_loss(agent, actual_policy, old_policy, ADV_err, action, pi_1, k=3):
    kldiv_loss = nn.KLDivLoss(reduction="batchmean")
    beta = 2
    omega = 4.0
    omega_12 = 2
    
    loss1 = loss_pg(actual_policy[0], pi_1, ADV_err, action)
    loss2 = loss_ppo(kldiv_loss, actual_policy, old_policy, beta, omega, k)
    loss3 = loss_casc(kldiv_loss, actual_policy, old_policy, omega, omega_12, k)
    loss = loss1 + loss2 + loss3
    #print(f"Total loss value: {loss}\n")

    return loss


def loss_pg(actual_policy, pi_1, ADV_err, action):
    loss = torch.tensor((actual_policy[action]/pi_1[action])*ADV_err, requires_grad=True)
    #print(f"PG loss value: {loss}")
    return loss

def loss_ppo(kldiv_loss, actual_policy, old_policy, beta, omega, k=3):
    # k -----> set of policies used in Synaptic Consolidation
    loss = 0
    for i in range(0, k):
        loss += -beta*(omega**(k))*kldiv_loss(old_policy[i], actual_policy[i]) 
    # KLDiv(output of model, observated experience) ---> the output of the model must be computed with logSoftmax, not Softmax
    #print(f"PPO loss value: {loss}")
    return loss

def loss_casc(kldiv_loss, actual_policy, old_policy, omega, omega_12, k=3):
    loss = -(omega_12)*kldiv_loss(old_policy[1], actual_policy[0]) 
    for i in range(1, k-1):
        loss += -(omega)*kldiv_loss(old_policy[i-1], actual_policy[i]) - kldiv_loss(old_policy[i+1], actual_policy[i])
    loss += -(omega)*kldiv_loss(old_policy[k-2], actual_policy[k-1]) - kldiv_loss(old_policy[k-1], actual_policy[k-1])
    #print(f"CASC loss value: {loss}")
    return loss