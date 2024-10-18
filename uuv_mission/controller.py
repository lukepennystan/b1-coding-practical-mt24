class PDController:
    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.previous_error = 0.0

    def compute_action(self, error: float) -> float:
        # Compute the derivative of the error
        derivative = error - self.previous_error
        
        # Compute the control action
        action = self.kp * error + self.kd * derivative
        
        # Update the previous error
        self.previous_error = error
        
        return action