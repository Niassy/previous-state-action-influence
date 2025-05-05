import gymnasium
import highway_env
from matplotlib import pyplot as plt
from q_learning import QLearningSimple

map_action = {
    0:1,
    1:3,
    2:4 
}

# Our code
Q = QLearningSimple()
env = gymnasium.make('highway-v0', render_mode='rgb_array')


INFLUENCE_FACTOR = 0.9

num_episode = 100

rewards_per_episode = []

number_hit = 0

use_influence = False
for num in range(num_episode):

    if num == num_episode // 2:
        print("Inference")

    obs,info = env.reset()

    distance = info["dist_x"]
    speed = info["vehicle1_speed"]

    steps = 0
    done = False
    total_reward = 0

    dict_prev_state_action = {}
    memory_array = []

    hit = False

    while not done:

        reward = 0

        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        #obs, reward, done, truncated, info = env.step(action)

        #print("dist x = ",info["dist_x"] )
        #print("dist y = ",info["dist_y"] )

        #print("speed ",str(speed))

        state = Q.discretize_state(distance, speed)

        
        if num < num_episode // 2:
            action =Q.choose_action(state)
        else:
            action =Q.choose_action(state,True)

        real_action = map_action[action] 

        obs, _, done, truncated, info = env.step(real_action)
        env.render()

        if info.get("crashed",True):
            reward = - 10
            done = True
            hit = True
            number_hit+=1

        total_reward+=reward
        next_dist = info["dist_x"]
        next_speed = info["vehicle1_speed"]

        if next_speed > speed:
            reward+=1

        next_state = Q.discretize_state(next_dist, next_speed )
        Q.learn_v2(state,action,reward,next_state)

        distance = next_dist
        speed = next_dist

        steps+=1

        if steps > 50:
            done = True

        memory = (state, action, reward, next_state, done)
        
        key = next_state
        val = (state,action)

            # add the key if a new entry
            # add time for key
        if key not in dict_prev_state_action.keys():
            #print("key ", key, " not in dict ")
            dict_prev_state_action[ key ] = val
            

    # at the end of the episode
    rewards_per_episode.append(total_reward)

    if hit and use_influence:
            #print("memory array  ", memory_array)
        visited = []
        #print("  ------------influence  start in memory ----------- ")

        reverse_array = []
        for mem in memory_array:
            state, action, reward, next_state, done = mem

            influence = 0
            if reward < 0:
                influence = INFLUENCE_FACTOR
            
            if state not in visited and  state in dict_prev_state_action:
                prev_state, prev_action = dict_prev_state_action[state]

                if prev_state[0] ==state[0]  and prev_state[1] == state[1]:
                    continue

                old_table = Q.q_table[prev_state][ prev_action]
                Q.q_table[prev_state][ prev_action]+=  reward * influence
                visited.append(state)
                reverse_array.append( (state,prev_state,prev_action))
                
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

            if loop >= 30:
                break


print("Hits = ",str(number_hit)+" / ",str(num_episode))

plt.plot(rewards_per_episode)
plt.title("Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()


