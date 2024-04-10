import numpy as np

class DeadReckoningEstimator:
    def __init__(self, initial_orientation, initial_velocity):
        self.orientation = initial_orientation  # Initial orientation (in radians)
        self.velocity = initial_velocity  # Initial velocity (in m/s)
        self.acceleration = (0, 0, 0)  # Initial acceleration (in m/s^2)
        self.angular_velocity = (0, 0, 0)  # Initial angular velocity (in rad/s)

    def update(self, gyro_data, accel_data, time_interval):
        # Gyro data: (gyro_x, gyro_y, gyro_z) in rad/s
        # Accel data: (accel_x, accel_y, accel_z) in m/s^2
        # Time interval: Time elapsed since last update (in seconds)

        # Update angular velocity using gyro data
        self.angular_velocity = gyro_data

        # Update orientation based on angular velocity
        self.orientation = tuple(
            [
                self.orientation[i] + self.angular_velocity[i] * time_interval
                for i in range(3)
            ]
        )

        # Update acceleration using accelerometer data
        self.acceleration = accel_data

        # Update velocity based on acceleration
        self.velocity = tuple(
            [
                self.velocity[i] + self.acceleration[i] * time_interval
                for i in range(3)
            ]
        )

    def get_orientation(self):
        return self.orientation

    def get_velocity(self):
        return self.velocity
    

if __name__ == "__main__":


    # Example usage:
    initial_orientation = (0, 0, 0)  # Initial orientation (in radians)
    initial_velocity = (0, 0, 0)  # Initial velocity (in m/s)
    estimator = DeadReckoningEstimator(initial_orientation, initial_velocity)

    # Example sensor data
    gyro_data = (0.1, 0.05, 0.03)  # Angular velocity (in rad/s)
    accel_data = (0.1, 0.2, 9.8)  # Acceleration (in m/s^2)
    time_interval = 0.1  # Time interval (in seconds)

    # Update the estimator with sensor data
    estimator.update(gyro_data, accel_data, time_interval)

    # Get the estimated orientation and velocity
    estimated_orientation = estimator.get_orientation()
    estimated_velocity = estimator.get_velocity()

    print("Estimated Orientation:", estimated_orientation)
    print("Estimated Velocity:", estimated_velocity)
