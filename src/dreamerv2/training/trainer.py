# File: dreamerv2/training/trainer.py

import numpy as np
import torch 
import torch.optim as optim
import torch.amp as amp
import os 

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer

from torch import save
from pathlib import Path
from dreamerv2.utils.pathing import ensure_dir

class Trainer(object):
    def __init__(
        self, 
        config,
        device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip

        self._model_initialize(config)
        self._optim_initialize(config)
        self.scaler = amp.GradScaler("cuda")              # ← initialize gradient scaler

    def collect_seed_episodes(self, env):

        # Unwrap VecToSingle if present to get the underlying vector env
        base_env = getattr(env, "env", env)
        # Reset returns (obs_batch, info_batch)
        reset_res = base_env.reset()
        obs_batch = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        # Collect seed_steps batches of random transitions
        for _ in range(self.seed_steps):
            # Sample one action per parallel env
            actions_b = base_env.action_space.sample()
            # Step vector env -> (obs_b, rew_b, term_b, trunc_b, info_b)
            obs_n_b, rew_b, term_b, trunc_b, _ = base_env.step(actions_b)
            done_b = np.logical_or(term_b, trunc_b)  # (N,) bool array
            # Add each sub-env's transition into the buffer
            for ob, a, r, d in zip(obs_batch, actions_b, rew_b, done_b):
                self.buffer.add(ob, a, float(r), bool(d))
            # Prepare next batch
            obs_batch = obs_n_b

    def train_batch(self, train_metrics):
        """ 
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)                         #t, t+seq_len 
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)                 #t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = torch.tensor(1-terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  #t-1 to t+seq_len-1

            # --- BEGIN FIX: mixed precision autocast for world model ---
            with amp.autocast(device_type="cuda"):  # autocast for mixed precision
               model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, \
               prior_dist, post_dist, posterior = \
                   self.representation_loss(obs, actions, rewards, nonterms)
           # --- END FIX ---
            
            self.model_optimizer.zero_grad()
            # --- BEGIN FIX: scale model loss for mixed precision ---
            self.scaler.scale(model_loss).backward()                                   # scale & backward
            self.scaler.unscale_(self.model_optimizer)                                 # unscale before clip
            grad_norm_model = torch.nn.utils.clip_grad_norm_(                          # clip grads
                get_parameters(self.world_list), self.grad_clip_norm)
            self.scaler.step(self.model_optimizer)                                     # step via scaler
            self.scaler.update()                                                       # update the scale
            # --- END FIX ---

                       # --- BEGIN FIX: mixed precision autocast for actor & critic ---
            with amp.autocast(device_type="cuda"):
                actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)
           # --- END FIX ---

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            # --- BEGIN FIX: scale actor & critic losses for mixed precision ---
            # Actor
            self.scaler.scale(actor_loss).backward()                                    # scale & backward actor
            self.scaler.unscale_(self.actor_optimizer)                                  # unscale before clip
            grad_norm_actor = torch.nn.utils.clip_grad_norm_(                           # clip actor grads
                get_parameters(self.actor_list), self.grad_clip_norm)
            self.scaler.step(self.actor_optimizer)                                      # actor step

            # Critic / Value
            self.scaler.scale(value_loss).backward()                                    # scale & backward value
            self.scaler.unscale_(self.value_optimizer)                                  # unscale before clip
            grad_norm_value = torch.nn.utils.clip_grad_norm_(                           # clip value grads
                get_parameters(self.value_list), self.grad_clip_norm)
            self.scaler.step(self.value_optimizer)                                      # value step

            self.scaler.update()  

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss']=np.mean(kl_l)
        train_metrics['reward_loss']=np.mean(reward_l)
        train_metrics['obs_loss']=np.mean(obs_l)
        train_metrics['value_loss']=np.mean(value_l)
        train_metrics['actor_loss']=np.mean(actor_l)
        train_metrics['prior_entropy']=np.mean(prior_ent_l)
        train_metrics['posterior_entropy']=np.mean(post_ent_l)
        train_metrics['pcont_loss']=np.mean(pcont_l)
        train_metrics['mean_targ']=np.mean(mean_targ)
        train_metrics['min_targ']=np.mean(min_targ)
        train_metrics['max_targ']=np.mean(max_targ)
        train_metrics['std_targ']=np.mean(std_targ)

        return train_metrics

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        
        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]+[self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)
            discount_arr = discount_arr.float() 

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)     

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }

        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms):

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        # Base world-model loss
        model_loss = (
            self.loss_scale['kl']     * div +
            self.loss_scale['reward'] * reward_loss +
            self.loss_scale['obs']    * obs_loss +
            self.loss_scale['discount'] * pcont_loss
        )

        # ─── Slot-Attention bias losses ───
        if self.config.pixel:
            # (1) get the last attention maps and spatial dims
            attn_flat = self.ObsEncoder.slot_attn.last_attn       # (B*(S+1), K, Hp*Wp)
            Hp, Wp     = self.ObsEncoder.slot_attn.last_Hp_Wp
            B, T       = obs.shape[0], obs.shape[1]
            attn       = attn_flat.view(B, T, -1, Hp, Wp)        # (B, T, K, Hp, Wp)

            # (2) spatial continuity (TV loss)
            tv = torch.abs(attn[:, :, :, 1:, :] - attn[:, :, :, :-1, :]).mean()
            tv = tv + torch.abs(attn[:, :, :, :, 1:] - attn[:, :, :, :, :-1]).mean()

            # (3) temporal smoothness
            m_t, m_tm1 = attn[:, 1:], attn[:, :-1]
            temp = torch.abs(m_t - m_tm1).mean()

            # (4) bounding box regularization via slot variance
            ys = torch.linspace(0, Hp - 1, Hp, device=attn.device).view(1,1,1,Hp,1)
            xs = torch.linspace(0, Wp - 1, Wp, device=attn.device).view(1,1,1,1,Wp)
            y_mean = (attn * ys).sum(dim=(3,4), keepdim=True)
            x_mean = (attn * xs).sum(dim=(3,4), keepdim=True)
            y_var  = (attn * (ys - y_mean)**2).sum(dim=(3,4), keepdim=True)
            x_var  = (attn * (xs - x_mean)**2).sum(dim=(3,4), keepdim=True)
            box_loss = (y_var + x_var).mean()

            # (5) combine with configurable weights
            lambda_box   = self.config.slot_attention.loss.box
            lambda_tv    = self.config.slot_attention.loss.tv
            lambda_temp  = self.config.slot_attention.loss.temporal
            model_loss = model_loss + lambda_box * box_loss + lambda_tv * tv + lambda_temp * temp

        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior


    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        # --- BEGIN FIX: unify dtypes (AMP) --------------------------------
        imag_reward   = imag_reward.float()
        imag_value    = imag_value.float()
        discount_arr  = discount_arr.float()
        imag_log_prob = imag_log_prob.float()
        policy_entropy = policy_entropy.float()
        # --- END FIX -------------------------------------------------------

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates) 
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss
            
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter=None, tag: str | None = None):
        """
        • iter=N  –→ models_000123.pth
        • tag='best'   –→ models_best.pth
        • tag='latest' –→ models_latest.pth
        """
        model_dir = ensure_dir(self.config.model_dir)

        if tag:
            fname = f"models_{tag}.pth"
        elif iter is not None:
            fname = f"models_{iter:06d}.pth"
        else:
            fname = "models_latest.pth"

        save_dict = self.get_save_dict()
        save(save_dict, model_dir / fname)  # ← swap args: obj first, then path so f is a valid file‐path
    
    def save_if_missing(self, tag: str = "best") -> Path:
        """
        Save the current weights as models_<tag>.pth *only if* that file
        does not exist yet.  Returns the concrete Path either way.
        """
        model_dir = ensure_dir(self.config.model_dir)
        ckpt = model_dir / f"models_{tag}.pth"
        if not ckpt.exists():
            self.save_model(tag=tag)
            print(f"[trainer] Saved new '{tag}' checkpoint → {ckpt}")
        return ckpt

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])
            
    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size 
    
        self.buffer = TransitionBuffer(config.capacity, obs_shape, action_size, config.seq_len, config.batch_size, config.obs_dtype, config.action_dtype)
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        print('\n Actor: \n', self.ActionModel)
        print('\n Critic: \n', self.ValueModel)
    
    def load_model(self, ckpt_path):
        """Load model checkpoint from given path."""
        saved_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.load_save_dict(saved_dict)

    def eval(self):
        """Set all model components to eval mode."""
        self.RSSM.eval()
        self.ObsEncoder.eval()
        self.ObsDecoder.eval()
        self.RewardDecoder.eval()
        self.ActionModel.eval()
        self.ValueModel.eval()
        self.TargetValueModel.eval()
        if hasattr(self, 'DiscountModel'):
            self.DiscountModel.eval()
