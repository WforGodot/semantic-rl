import torch

def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                discount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float
            ):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    # --- BEGIN FIX: unify dtypes to avoid Half/Float mismatches ---
    dtype = reward.dtype                              # ← base dtype from reward
    device = reward.device                            # ← keep same device
    lambda_tensor = torch.tensor(lambda_, dtype=dtype, device=device)  # ← cast lambda_ to tensor
    bootstrap = bootstrap.to(dtype)                   # ← ensure bootstrap matches dtype
    discount = discount.to(dtype)                    # ← ensure discount matches dtype
    # --- END FIX ---

    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_tensor)  # ← use tensor lambda for dtype consistency

    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_tensor * accumulated_reward  # ← use tensor lambda
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns
