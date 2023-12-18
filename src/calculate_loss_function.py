import torch
    

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

def loss_ppo( agent, actual_id_env, state, next_state, reward, done):
    
    return None

def loss_casc(agent):
    return None