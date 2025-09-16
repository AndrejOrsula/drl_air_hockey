import embodied


# Patch missing __len__ method in embodied.replay.selectors.Mixture
class Mixture(embodied.replay.selectors.Mixture):
    def __len__(self):
        return len(self.selectors[0])
