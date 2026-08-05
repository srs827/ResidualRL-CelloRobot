"""

Placeholder quality classifier for RL development.
All values are fake -- we just want to see if RL can converge. 
Models quality as a function of z-displacement from string baseline.
We will be using real version as a major component of the reward function
[0.0,1.0] -> higher = better sound quality

"""

import numpy as np


class ExSoundClassifier:
    """
    Simulates classifier output during RL development.

    Quality model:
        - Force near optimal -> high quality
        - Too much force -> raucous (quality drops)
        - Too little force -> flautando (quality drops)
    """

    # Optimal force per string
    # (fake values, means nothing)
    OPTIMAL_FORCE = {'A': 2.8, 'D': 3.1, 'G': 3.5, 'C': 4.0}
    SIGMA         = 1.2    # how quickly quality degrades from optimal
    NOISE_STD     = 0.04   # classifier noise

    def predict(
        self,
        audio_chunk,       # ignored here, real classifier will use it
        physical_params,   # np.ndarray (6,) -- uses force component
        string = 'A',      # single string
    ) -> float:
        """
        Returns quality score [0, 1], where the higher the better.
        """
        # Extract force from physical params
        # physical_params[0] = force_mean (from get_physical_params())
        force   = float(physical_params[0]) if physical_params is not None else 3.0
        optimal = self.OPTIMAL_FORCE.get(string, 3.0)

        # Gaussian quality model centered on optimal force
        quality = np.exp(-0.5 * ((force - optimal) / self.SIGMA) ** 2)
        quality = float(np.clip(
            quality + np.random.normal(0, self.NOISE_STD), 0.0, 1.0
        ))

        return quality