import pygame
import numpy as np
import random

import matplotlib.pyplot as plt

from fontTools.ttLib.ttVisitor import visit

from q_learning import QLearningSimple

# Pygame setup
pygame.init()

# Constants
GRID_SIZE = 5
CELL_SIZE = 64
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Reward system
REWARD_GOAL = 10
REWARD_OBSTACLE = -5
#REWARD_STEP = -0.1  # Small penalty for each step

REWARD_STEP = 0

# 0.35
INFLUENCE_FACTOR = 0.9

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate

# Speed and action parameters
MAX_SPEED = 10
ACCELERATION = 1  # Velocity change per action


ACTIONS = [  # Actions to change speed
    [0, 0],  # No change
    [-ACCELERATION, 0],  # Accelerate left
    [ACCELERATION, 0]  # Accelerate right
    #[0, -ACCELERATION],  # Accelerate up
    #[0, ACCELERATION],  # Accelerate down
]

# Initialize Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Q-learning with Continuous Speed")
clock = pygame.time.Clock()

# Agent setup
agent_pos = [0, 0]  # Initial position (x, y) in pixels
agent_speed = [0, 0]  # Initial speed (vx, vy)

# Goal and obstacle positions
goal_pos = [CELL_SIZE * (GRID_SIZE - 1), CELL_SIZE * (GRID_SIZE - 1)]  # Bottom-right corner
obstacle_pos = [CELL_SIZE * 4, CELL_SIZE * 2]  # Middle of the grid
obstacle_size = CELL_SIZE


# our code
Q = QLearningSimple()

#Q.load_model("gridworld_continuous_rl1.npy")



def draw_grid():
    """Draw the grid, goal, obstacle, and agent."""
    screen.fill(WHITE)

    # Draw grid lines
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

    # Draw goal
    pygame.draw.rect(screen, GREEN, (goal_pos[0], goal_pos[1], CELL_SIZE, CELL_SIZE))

    # Draw obstacle
    pygame.draw.rect(screen, RED, (obstacle_pos[0], obstacle_pos[1], obstacle_size, obstacle_size))

    # Draw agent
    pygame.draw.rect(screen, BLUE, (agent_pos[0], agent_pos[1], CELL_SIZE, CELL_SIZE))


def is_collision(pos1, size1, pos2, size2):
    """Check if two rectangles collide."""
    return (
            pos1[0] < pos2[0] + size2 and
            pos1[0] + size1 > pos2[0] and
            pos1[1] < pos2[1] + size2 and
            pos1[1] + size1 > pos2[1]
    )


#def get_reward(state,prev_pos = (-1,-1),episode_end ,distance ):
def get_reward(state, prev_pos, episode_end, dist):

    """Return the reward for the agent's current state."""
    if is_collision(state, CELL_SIZE, obstacle_pos, obstacle_size):
        return REWARD_OBSTACLE
    if is_collision(state, CELL_SIZE, goal_pos, CELL_SIZE):
        return REWARD_GOAL

    # also give bad reward when agent is far from goal
     #if episode_end:
        #if dist > 20:
            #return -1

    # if getting gloser
    if state[0] - prev_pos[0]  > 0 and prev_pos[0] != -1 and prev_pos[0] != -1:
        return 0.1

    return REWARD_STEP


def get_state(pos):
    """Convert continuous agent position into a discrete grid state."""
    x = pos[0] // CELL_SIZE
    y = pos[1] // CELL_SIZE
    return int(x), int(y)


def choose_action(state):
    """Choose an action using an epsilon-greedy policy."""
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)  # Explore
    else:
        return np.argmax(q_table[state[0], state[1]])  # Exploit


def train_agent(episodes):
    """Train the agent using Q-learning."""

    Q.load_model("gridworld_continuous_rl1.npy")

    dict_prev_state_action = {}

    memory_array = []

    rewards_training_episode = []
    global agent_pos, agent_speed

    for episode in range(episodes):
        print("-------- episode",episode," begins --------")
        agent_pos = [0, CELL_SIZE * 2]  # Reset position to top-left corner
        agent_speed = [0, 0]  # Reset speed to zero
        done = False
        hit =  False


        v = [2,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             1,0]

        ind = 0

        prev = []

        timer =  200


        total_rew = 0

    
        while not done:

            prev_pos = (agent_pos[0],agent_pos[1])
            prev_speed =( agent_speed[0], agent_speed[1])

            #prev_pos = agent_pos
            #prev_speed = agent_speed

            #state = get_state(agent_pos)  # Current grid state
            #action = choose_action(state)  # Choose action

            distance =  abs( obstacle_pos[0] - ( agent_pos[0] + CELL_SIZE) )
            speed = agent_speed[0]
            state =    Q.discretize_state(distance, speed)
            action =Q.choose_action(state)

            # extreme case of only accelerating
            #action = 2
            #action = v[ind]
            #ind+=1

            #if ind >len(v) - 1:
                #ind = len(v) - 1

            if action == 1:  # he can not go left
                if agent_speed[0] <=0:
                    action = 0

            acceleration = ACTIONS[action]  # Get acceleration vector

            # Update agent's speed
            agent_speed[0] += acceleration[0]
            agent_speed[1] += acceleration[1]

            # Clamp speed to maximum limits
            agent_speed[0] = max(-MAX_SPEED, min(agent_speed[0], MAX_SPEED))
            agent_speed[1] = max(-MAX_SPEED, min(agent_speed[1], MAX_SPEED))


            # Update agent position based on speed
            agent_pos[0] += agent_speed[0]
            agent_pos[1] += agent_speed[1]

            # Keep agent within bounds
            agent_pos[0] = max(0, min(agent_pos[0], SCREEN_WIDTH - CELL_SIZE))
            agent_pos[1] = max(0, min(agent_pos[1], SCREEN_HEIGHT - CELL_SIZE))

            #next_state = get_state(agent_pos)  # Get next grid state
            next_dist =  abs( obstacle_pos[0]- (agent_pos[0 ] + CELL_SIZE )  )
            next_speed = agent_speed[0]

            next_state = Q.discretize_state(next_dist, next_speed )


            reward = get_reward(agent_pos,prev_pos,done,next_dist)  # Get reward for next state

            total_rew+=reward

            # Update Q-value
            #old_value = q_table[state[0], state[1], action]
            #next_max = np.max(q_table[next_state[0], next_state[1]])
            #q_table[state[0], state[1], action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

            Q.learn_v2( state, action, reward, next_state)

            #print("state = ", state, " action = ", action, "reward = ", reward, "distance = ", distance,
                  #"agent_pos ", prev_pos[0],"obs = ",obstacle_pos[0], " agent_speed = ", prev_speed[0])

            #print( "state = ",state ," action = ",action,"reward = " ,reward, "distance = ",distance ,"agent_old_pos ", agent_pos[0]," agent_speed = ",agent_speed[0])

            # Check for terminal conditions
            if is_collision(agent_pos, CELL_SIZE, goal_pos, CELL_SIZE):
                done = True
            elif is_collision(agent_pos, CELL_SIZE, obstacle_pos, obstacle_size):

                print("Agent hit an obstacle. agent_pos = ",agent_pos [0] )
                done = True
                hit = True

            timer-=1
            if timer<=0:
                done = True

            memory = (state, action, reward, next_state, done)
            memory_array.append(memory)

            # record the previous state action
            key = next_state
            val = (state,action)

            # add the key if a new entry
            # add time for key
            if key not in dict_prev_state_action.keys():
                #print("key ", key, " not in dict ")
                dict_prev_state_action[ key ] = val
            #else:

                #print("key ",key ," in dict ")


            # otherwise update the q value
            #else:
             #   val
              #  Q.q_table[state][action]

            # reward the previous state


            # Render the environment
            draw_grid()
            pygame.display.flip()
            clock.tick(FPS)


        rewards_training_episode.append(total_rew)
        #total_rew = 0

        print(f"Episode {episode + 1} complete. Agent reached goal or obstacle.")

        if hit:
            #print("memory array  ", memory_array)
            visited = []
            print("  ------------influence  start in memory ----------- ")

            reverse_array = []
            for mem in memory_array:
                state, action, reward, next_state, done = mem

                # it mean the action was bad
                # we must also reward the previous stat
                influence = 0
                if reward < 0:
                    influence = INFLUENCE_FACTOR
                #influence = 0.1

                if state not in visited and  state in dict_prev_state_action:
                    prev_state, prev_action = dict_prev_state_action[state]

                    if prev_state[0] ==state[0]  and prev_state[1] == state[1]:
                        continue

                    old_table = Q.q_table[prev_state][ prev_action]
                    Q.q_table[prev_state][ prev_action]+=  reward * influence

                    #print("state = ",state ,"prev_state = ",prev_state,"prev_action = ",prev_action, "old = ",old_table, " table = ", Q.q_table[prev_state][ prev_action])

                    visited.append(state)
                    reverse_array.append( (state,prev_state,prev_action))
                    #Q.q_table[prev_state, prev_action]+=  np.min(Q.q_table[state])

            r = 0

            #print("  ------------influence  start in reverse ----------- ")

            loop = 0
            for comp in reversed( reverse_array):
                influence = INFLUENCE_FACTOR
                s, s_prev, act_prev = comp
                old_table = Q.q_table[s_prev][act_prev]
                Q.q_table[s_prev] [act_prev] += r * influence

                if Q.q_table[s_prev] [act_prev]<=-10:
                    Q.q_table[s_prev][act_prev] = - 10
                r = Q.q_table[s_prev] [act_prev]

                loop+=1

                if loop >= 6:
                    break

                #print("state = ", s, "prev_state = ", s_prev, "prev_action = ",
                      #act_prev, "old = ",old_table, " r = ",r," table = ", Q.q_table[s_prev][act_prev])

        # recursively reward all prev state
        #for key, val in dict_prev_state_action.items():
             #next_state = key
             #state, action = val
             #Q.q_table[state][action] +=0


            #val = (state, action)

    Q.save_model("gridworld_continuous_rl2.npy")

    plt.plot(rewards_training_episode)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


def main():
    training_episodes = 10  # Number of training episodes

    print("Training the agent...")
    train_agent(training_episodes)

    print("Training complete. Testing the agent...")

    global agent_pos, agent_speed

    agent_pos = [0, CELL_SIZE * 2]  # Reset position to top-left corner

    #agent_pos = [0, 0]  # Reset agent position for testing
    agent_speed = [0, 0]
    done = False

    Q.load_model("gridworld_continuous_rl2.npy")


    Q.epsilon =0.1
    rewards_per_episode = []
    success_per_episode = []
    for i in range(20):
        print("episode = ", i)
        t = 200
        done = False
        agent_pos = [0, CELL_SIZE * 2]  # Reset position to top-left corner
        agent_speed = [0, 0]
        total_reward = 0
        while not done:

            print("t = ",t)

            prev_pos = (agent_pos[0], agent_pos[1])
            prev_speed = (agent_speed[0], agent_speed[1])

            distance = abs(obstacle_pos[0] -( agent_pos[0] + CELL_SIZE) )
            speed = agent_speed[0]
            state = Q.discretize_state(distance, speed)
            #action = Q.choose_action(state)

            action =  np.argmax(Q.q_table[state])  # Exploit



            if action == 1:  # he can not go left
                if agent_speed[0] <=0:
                    action = 0

            #print(action)
            acceleration = ACTIONS[action]  # Get acceleration vector

            # Update agent speed
            agent_speed[0] += acceleration[0]
            agent_speed[1] += acceleration[1]

            # Clamp speed to maximum limits
            agent_speed[0] = max(-MAX_SPEED, min(agent_speed[0], MAX_SPEED))
            agent_speed[1] = max(-MAX_SPEED, min(agent_speed[1], MAX_SPEED))

            # Update agent position
            agent_pos[0] += agent_speed[0]
            agent_pos[1] += agent_speed[1]

            # Keep agent within bounds
            agent_pos[0] = max(0, min(agent_pos[0], SCREEN_WIDTH - CELL_SIZE))
            agent_pos[1] = max(0, min(agent_pos[1], SCREEN_HEIGHT - CELL_SIZE))

            next_dist = abs(obstacle_pos[0] - (agent_pos[0] + CELL_SIZE))

            reward = get_reward(agent_pos,prev_pos,done,next_dist)  # Get reward for next state

            # Check for terminal conditions
            if is_collision(agent_pos, CELL_SIZE, goal_pos, CELL_SIZE):
                print("Agent reached the goal!")
                done = True
            elif is_collision(agent_pos, CELL_SIZE, obstacle_pos, obstacle_size):
                print("Agent hit an obstacle.")
                done = True

            t-=1

            if t <= 0:
                done = True
                print("episode success")

            print("state = ", state, " action = ",action," d = ",distance," speed = ",speed,"table = ",Q.q_table[state] )

            total_reward+=reward

            # Render the environment
            draw_grid()
            pygame.display.flip()
            clock.tick(FPS)

        # at the end of the episode
        rewards_per_episode.append(total_reward)


    plt.plot(rewards_per_episode)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

if __name__ == "__main__":
    main()
    pygame.quit()
