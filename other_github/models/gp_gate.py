import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

class GPGate:
    '''
    Gaussian process (GP) gate that acts to map out a safe bow force space.
    '''
    def __init__(self, training_data: np.ndarray, uncertainty_threshold: float = 0.1):
        """
        training_data: 2D array of shape (n_samples, 12)
                       each row is a concatenated [force_6dof, tcp_6dof] vector
                       collected during warm-up. No audio scores needed here.
        
        uncertainty_threshold: sigma below which an action is approved.
                               Start low (0.1) to be conservative early in training.
        """

        self.training_data = list(training_data)  # list b/c can use .append()
        self.threshold = uncertainty_threshold

        # standard gaussian process (gp) implementation
        self.scaler = StandardScaler()
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=False
        )

        self.fit() # fit immediately after initializing with default training data

    def fit(self):
        """
        Retrain the GP on all accumulated training data.
        Called once at initialization on warm-up data,
        then periodically during training as new real-robot observations arrive.
        """

        # Convert list of vectors to 2D numpy array, shape (n_samples, 12)
        X = np.array(self.training_data)
        y = np.zeros(len(X)) # doesn't matter (only care about one-dimensional norm)!

        X_scaled = self.scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)

    def add_observation(self, force: np.ndarray, tcp: np.ndarray):
        """
        Add a new real-robot observation to the training set.
        
        force: (6,) array
        tcp:   (6,) array
        """

        observation = np.concatenate([force, tcp])
        self.training_data.append(observation)

    def query(self, force: np.ndarray, tcp: np.ndarray) -> float:
        """
        Query the GP for uncertainty at a proposed (force, TCP) action.
        Returns sigma — the standard deviation of the GP's prediction.
        
        Low sigma  = familiar region = safe to execute
        High sigma = unfamiliar region = reject
        
        force: (6,) array
        tcp:   (6,) array
        """

        state = np.concatenate([force, tcp])  # shape (12,)
        state_scaled = self.scaler.transform(state.reshape(1, -1)) # make 2d arr

        _, std = self.gp.predict(state_scaled, return_std=True) # type: ignore

        # std is a (1,) array, extract the scalar value
        return float(std[0])

    def is_approved(self, force: np.ndarray, tcp: np.ndarray) -> bool:
        return self.query(force, tcp) < self.threshold

    def anneal_threshold(self, new_threshold = 0.1):
        self.threshold = new_threshold