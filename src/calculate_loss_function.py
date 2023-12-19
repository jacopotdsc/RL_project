import torch
import torch.nn as nn

def Dkl(actual_policy, old_policy):
    '''
        env 1
        [ 0.3, 0.1, 0.2, 0.4 ]

            training -> backward(loss) -> new weight

        [ 0.2, 0.5, 0.25, 0.05 ]


        0.3 * log ( 0.3/0.2  ) -> v1
        0.1 * log ( 0.1/0.5  ) -> v2 

        distance = v1 + v2 ...
    '''

    actual_policy = actual_policy.clamp(min=1e-9)
    old_policy = old_policy.clamp(min=1e-9)

    distance = torch.sum( actual_policy * torch.log( actual_policy/old_policy ) )

    return distance

def total_loss():
    loss = loss_pg() + loss_ppo() + loss_casc()

    return loss


def loss_pg(agent):
    return None

def loss_ppo(agent , actual_policy, old_policy, beta, omega, k=1):
    # k -----> set of policies used in Synaptic Consolidation

    
    kldiv_loss = nn.KLDivLoss(reduction="batchmean")

    #print(f"ppo, actual policy: {actual_policy}")
    #print(f"ppo, old policy: {old_policy}")
    #print(f"ppo, kl div: {kldiv_loss(actual_policy, old_policy)}")

    result = -(1/k)*beta*(omega**(k-1))*kldiv_loss(actual_policy, old_policy) #.clone().detach().requires_grad_(True)
    
    #print(f"ppo, total loss value: {result}")
    return result

def loss_casc(agent):
    return None