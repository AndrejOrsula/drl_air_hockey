from dreamerv3.embodied.envs.from_gym import FromGym


class EmbodiedChallengeWrapper(FromGym):
    def __init__(self, env, obs_key="vector", act_key="action"):
        self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
