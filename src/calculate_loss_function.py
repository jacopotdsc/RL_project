import torch
import torch.nn as nn

def Dkl(actual_policy, old_policy):

    actual_policy = torch.mean(actual_policy.type(torch.FloatTensor), axis=0)
    old_policy    = torch.mean(old_policy.type(torch.FloatTensor), axis=0)
    distance = torch.sum( actual_policy * torch.log( actual_policy/old_policy ) )
   
    return distance

def total_loss():
    loss = loss_pg() + loss_ppo() + loss_casc()

    return loss


def loss_pg(agent):
    return None

def loss_ppo(actual_id_env, actual_policy, old_policy, beta, omega, k=1):
    # k -----> set of policies used in Synaptic Consolidation

    
    kldiv_loss = nn.KLDivLoss(reduction="batchmean")

    #print(f"ppo, actual policy: {actual_policy}")
    #print(f"ppo, old policy: {old_policy}")
    #print(f"ppo, kl div: {kldiv_loss(actual_policy, old_policy)}")

    #result = -(1/k)*beta*(omega**(k-1))*kldiv_loss(actual_policy, old_policy) #.clone().detach().requires_grad_(True)
    my_beta = 0.7
    distance = my_beta * kldiv_loss(actual_policy, old_policy)
    #print(f"id: {actual_id_env}, loss value: {distance}, mean act {torch.mean(actual_policy)}[{len(actual_policy)}],mean old {torch.mean(old_policy)}[{len(old_policy)}],  type: {type(distance)}\n")
    print("id: {}, loss value: {:.3f}, mean act {:.3f}[{}], mean old {:.3f}[{}], type: {}\n".format( actual_id_env, distance, torch.mean(actual_policy), len(actual_policy),torch.mean(old_policy), len(old_policy), type(distance).__name__))
    
    #print(f"ppo, total loss value: {distance}")
    return distance

def loss_casc(agent):
    return None