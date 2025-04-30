import numpy as np
import itertools

class QLearningSimple:
    def __init__(self, n_distance_bins=20,n_speed_bins=10, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.9):
        # Discretize distance features into bins
        self.distance_bins = np.linspace(0, 50, n_distance_bins)
        self.speed_bins = np.linspace(0, 10, n_speed_bins)

        self.actions = list(range(n_actions))  # Idle, left, right, accelerate, decelerate

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table with zeros
        # Dimensions: distance_x_bins * left_lane_bins * right_lane_bins * 2 (getting_closer)
        self.q_table = np.zeros((n_distance_bins, n_speed_bins, n_actions))

        #print(self.q_table)

        print(self.distance_bins)

        all_states = list(itertools.product(self.distance_bins, self.speed_bins))

        # Convert to a numpy array if needed
        all_states_array = np.array(all_states)

        print("Total number of states:", len(all_states))
        #print("All possible states:\n", all_states_array)

    def discretize_state(self, distance, speed):
        # Discretize continuous distances into bins
        dist_bin = np.digitize(distance, self.distance_bins) - 1
        speed_bin = np.digitize(speed, self.speed_bins) - 1

        # Clip to ensure values fall within range
        dist_bin = np.clip(dist_bin, 0, len(self.distance_bins) - 1)
        speed_bin = np.clip(speed_bin, 0, len(self.speed_bins) - 1)

        # Return tuple representing discrete state
        return (
        dist_bin, speed_bin)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def get_best_action_for_state_excluding_zero(self, state):
        # Get Q-values for all actions in the specified state
        action_values = self.q_table[state]

        # Exclude action `0` and find t.he action with the highest Q-value
        best_action = np.argmax(action_values[1:]) + 1  # +1 adjusts index since we skip action 0

        # Return the best action and its Q-value
        return best_action, action_values[best_action]

    def get_best_action_for_state_excluding_two(self, state):
        # Get Q-values for all actions in the specified state
        action_values = self.q_table[state]

        # Exclude action `2` by setting its value to a very low number (e.g., -inf)
        action_values_excluding_two = np.copy(action_values)
        action_values_excluding_two[2] = -np.inf

        # Find the action with the highest Q-value excluding action `2`
        best_action = np.argmax(action_values_excluding_two)

        # Return the best action and its Q-value
        return best_action, action_values[best_action]

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def learn_v2(self,state,action,reward,next_state):
        self.update_q_value(state, action, reward, next_state)

    def learn(self, distance, speed, action,
              reward, next_distance, next_speed):
        # Discretize current and next states
        state = self.discretize_state(distance, speed)
        next_state = self.discretize_state(next_distance, next_speed)

        # Update Q-table
        self.update_q_value(state, action, reward, next_state)


    def save_model(self, file_path):
        # Save the Q-table to a file
        np.save(file_path, self.q_table)
        print(f"Q-table saved to {file_path}")

    def load_model(self, file_path):
        # Load the Q-table from a file
        self.q_table = np.load(file_path)
        print(f"Q-table loaded from {file_path}")


    def printing(self):
        print(" shape "+str(self.q_table.shape[0]))

        # Function to iterate through all the states and actions in the Q-table

    """
    def iterate_q_table(self):
        # Iterate over all 125 states (n_distance_bins^3)
        for dist_x_bin in range(self.q_table.shape[0]):  # 5 bins for distance_x
            for dist_left_bin in range(self.q_table.shape[1]):  # 5 bins for left lane
                for dist_right_bin in range(self.q_table.shape[2]):  # 5 bins for right lane
                    for action in range(self.q_table.shape[3]):  # 5 actions
                        q_value = self.q_table[dist_x_bin, dist_left_bin, dist_right_bin, action]
                        print(
                            f"State ({dist_x_bin}, {dist_left_bin}, {dist_right_bin}), Action {action}, Q-value: {q_value}")
"""
    """
    def state_to_index(self, state):
        # Map 3D state (distance_x_bin, distance_left_bin, distance_right_bin) to a 1D index
        dist_bin, speed_bin = state
        return dist_bin * (self.q_table.shape[1] * self.q_table.shape[2]) + dist_left_bin * self.q_table.shape[
            2] + dist_right_bin
            
    """
    def collect_visited_state_indices(self):
        visited_state_indices = []
        # Iterate over all states
        l = 0
        for dist_x_bin in range(self.q_table.shape[0]):
            for dist_left_bin in range(self.q_table.shape[1]):
                for dist_right_bin in range(self.q_table.shape[2]):
                    # Check if any action for this state has a Q-value > 0
                    if np.any(self.q_table[dist_x_bin, dist_left_bin, dist_right_bin] != 0):
                        state = (dist_x_bin, dist_left_bin, dist_right_bin)
                        visited_state_indices.append(self.state_to_index(state))  # Add the 1D index
                        l+=1
        print("size visited states: " +str(l))
        return visited_state_indices


    """
    def get_state_index(self, state):
        # Convert state tuple to a unique index
        return state[0] * (self.q_table.shape[1] * self.q_table.shape[2]) + \
               state[1] * self.q_table.shape[2] + \
               state[2]
"""