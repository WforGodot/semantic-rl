import numpy as np
import torch 
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder

class Evaluator(object):
    '''
    used this only for minigrid envs
    '''
    def __init__(
        self, 
        config,
        device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
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

        if config.pixel:
                self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
                self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device).eval()
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])

    def eval_saved_agent(self, env, model_path):
        self.load_model(self.config, model_path)
        eval_episode = self.config.eval_episode
        eval_scores = []    
        for e in range(eval_episode):
            reset_res   = env.reset()
            obs         = reset_res[0] if isinstance(reset_res, tuple) else reset_res
            if isinstance(obs, np.ndarray) and obs.ndim == 4:     # vector env
                obs = obs[0]
            score = 0                                             # keep counter
            done = False
            prev_rssmstate = self.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    # convert to (1, C, H, W) tensor exactly as in training
                    tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    tensor_obs = tensor_obs.unsqueeze(0)
                    embed = self.ObsEncoder(tensor_obs)    
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    action, _ = self.ActionModel(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
            # step returns batched results: (obs_batch, rew_batch, term_batch, trunc_batch, info)
            next_obs_batch, rew_batch, term_batch, trunc_batch, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            done = bool(term_batch[0] or trunc_batch[0])
            reward = float(rew_batch[0])
            if self.config.eval_render:
                env.render()
            score += reward
            obs = next_obs_batch[0]
            eval_scores.append(score)
        print('average evaluation score for model at ' + model_path + ' = ' +str(np.mean(eval_scores)))
        env.close()
        return np.mean(eval_scores)
