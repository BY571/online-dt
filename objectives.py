import torch

def get_loss_function(stochastic, planning):
    if stochastic and planning:
        return stoch_pred
    elif stochastic and not planning:
        return stoch_action_pred
    elif not stochastic and planning:
        return deter_pred
    elif not stochastic and not planning:
        return deter_action_pred
    else:
        raise NotImplementedError

def stoch_action_pred(
    a_hat_dist,
    a,
    attention_mask,
    entropy_reg,
    s_hat_dist,
    s_target,
    r_hat_dist,
    r_target
):
    """
    Stochastic loss function for only action prediction
    """
    # a_hat is a SquashedNormal Distribution
    log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
    entropy = a_hat_dist.entropy().mean()

    # Placeholders as the stoch losses need to have the same outputs
    state_entropy = torch.zeros_like(log_likelihood)
    state_log_likelihood = torch.zeros_like(log_likelihood)
    reward_entropy = torch.zeros_like(log_likelihood)
    reward_loglikelihood = torch.zeros_like(log_likelihood)
    
    loss = -(log_likelihood + entropy_reg * entropy)

    return (
        loss,
        -log_likelihood,
        entropy,
        state_entropy,
        state_log_likelihood,
        reward_entropy,
        reward_loglikelihood
    )

def stoch_pred(
    a_hat_dist,
    a_target,
    attention_mask,
    entropy_reg,
    s_hat_dist,
    s_target,
    r_hat_dist,
    r_target
):
    """
    Stochastic loss function for action, state, reward predictions
    """
    # a_hat is a SquashedNormal Distribution
    log_likelihood = a_hat_dist.log_likelihood(a_target)[attention_mask > 0].mean()
    entropy = a_hat_dist.entropy().mean()
    
    # state prediction error
    state_log_likelihood = s_hat_dist.log_prob(s_target).sum(axis=2)[attention_mask > 0].mean()
    state_entropy = s_hat_dist.entropy().mean()
    
    # reward prediction
    reward_log_likelihood = r_hat_dist.log_prob(r_target).sum(axis=2)[attention_mask > 0].mean()
    reward_entropy = r_hat_dist.entropy().mean()

    loss = -(log_likelihood + entropy_reg * entropy)# + state_log_likelihood + entropy_reg * state_entropy)

    return (
        loss,
        -log_likelihood,
        entropy,
        state_entropy,
        state_log_likelihood,
        reward_entropy,
        reward_log_likelihood
    )

def deter_pred(a_hat, a, s_hat, s, r_hat, r, attention_mask):
    """
    Deterministic loss function for action, state and reward prediction
    """
    action_loss = torch.mean((a_hat - a)**2) 
    state_loss = torch.mean((s_hat - s)**2)
    reward_loss = torch.mean((r_hat - r)**2)

    loss = action_loss + state_loss + reward_loss
    return (loss, action_loss, state_loss, reward_loss)

def deter_action_pred(a_hat, a, s_hat, s, r_hat, r, attention_mask):
    """
    Deterministic loss function for only action prediction
    """
    action_loss = torch.mean((a_hat - a)**2)
    # Place holder as deter loss output is always the same
    state_loss = torch.zeros_like(action_loss)
    reward_loss = torch.zeros_like(action_loss)

    loss = action_loss + state_loss + reward_loss
    return (loss, action_loss, state_loss, reward_loss)