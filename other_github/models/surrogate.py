import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class SurrogateModel:
    def __init__(self, training_data: np.ndarray, training_scores: np.ndarray):
        """
        training_data: 2D array of shape (n_samples, 12)
                       each row is a concatenated [force_6dof, tcp_6dof] vector
                       collected during warm-up.
        
        training_scores: 1D array of shape (n_samples,)
                            gotten from CNN sound classification model
        """

        self.training_data = training_data
        self.training_scores = training_scores
        self.predicted_score = 0

        # standard MLP regressor implementation
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )

        self.fit() 

    def fit(self):
        """
        Train the MLP on the provided training data and scores.
        Called once at initialization on warm-up data,
        then periodically during training as new real-robot observations arrive.
        """

        if len(self.training_data) < 5 or len(self.training_scores) < 5:
            print("not enough data to train")
            return 
        
        X = self.training_data
        y = self.training_scores

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        try: 
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            print("Surrogate model trained sucessfully!")
        except Exception as e:
            print(f"error fitting surrogate model: {e}")
            self.is_fitted = False

    def add_observation(self, force: np.ndarray, tcp: np.ndarray, score: float):
        '''
        adds new force & tcp collected to training data, allows us to retrain surrogate as needed
        force is (6,) array, tcp is (6,) array, score is scalar
        '''
        self.training_data = np.vstack([self.training_data, np.concatenate([force, tcp])])
        self.training_scores = np.concatenate([self.training_scores, [score]])

    def predict_score(self, force: np.ndarray, tcp: np.ndarray) -> float:
        '''
        predict the sound score for a proposed (force, tcp) action
        returns a scalar score
        '''
        if not self.is_fitted:
            print("model not fitted yet, returning default score of 0")
            return 0
        
        observation = np.concatenate([force, tcp]).reshape(1, -1)
        observation_scaled = self.scaler.transform(observation)
        self.predicted_score = self.model.predict(observation_scaled)[0]
        return self.predicted_score
    
    def predict_reward(self):
        '''
        for now, keep reward simple :)
        '''
        return self.predicted_score * 1.0
    