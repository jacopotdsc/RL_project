import torch

def Dkl(actual_policy, old_policy):

    actual_policy = actual_policy.clamp(min=1e-9)
    old_policy = old_policy.clamp(min=1e-9)

    distance = torch.sum( actual_policy * torch.log( actual_policy/old_policy ) )

    return distance

def total_loss():
    loss = loss_pg + loss_ppo + loss_casc

    return loss


def loss_pg():
    return

def loss_ppo():
    return

def loss_casc():
    return