'''
Example of Q-learning

Finding the path from point A to point B in case of a 2d grid
    Loc_inp = ['L1', 'L2', 'L3',
               'L4', 'L5', 'L6',
               'L7', 'L8', 'L9']

Calculating Temporal difference: TD(s,a) = R(s,a) + gamma * Q(s',a') - Q(s,a)
Updating the q-value for a state using Bellman equation
Q(s,a) = Q(s,a) + alpha*TD(s,a), where alpha is the learning rate

The learning rate defines for, how fast a robot adapts to the changes in the environment
'''

# Importing the libraries
import numpy as np

def get_optimal_route(start_location, end_location):

    # Initializing the equation parameters where gamma is the discount
    # factor and alpha is the learning rate
    gamma = 0.75
    alpha = 0.9

    # Location input
    Loc_inp = ['L1', 'L2', 'L3',
               'L4', 'L5', 'L6',
               'L7', 'L8', 'L9']

    location_to_state = {}

    # Creating a table for location input in terms of a state
    for i in range(len(Loc_inp)):
        if Loc_inp[i] in location_to_state:
            continue
        else:
            location_to_state[Loc_inp[i]] = i

    # Define the list of actions
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Initializing a 2d matrix for the reward based on the locations
    # which are directly reachable from the current state of the robot
    rewards = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    # Mapping indices to locations
    state_to_location = dict((state, location) for location, state in location_to_state.items())

    # Copying reward matrix to a new_matrix
    rewards_new = np.copy(rewards)

    # Get the ending state corresponding to given ending location
    ending_state = location_to_state[end_location]

    # Setting the priority of the given ending state to the highest one
    rewards_new[ending_state,ending_state] = 999

    # Initializing Q-values
    Q = np.array(np.zeros([9,9]))

    # Q-Learning process
    for i in range(1000):

        # Picking a state randomly
        current_state = np.random.randint(0,9)

        # For traversing through the neighbor locaions in the maze
        playable_actions = []

        # Iterate through the new rewards matrix and get the actions > 0
        for j in range(9):
            if rewards_new[current_state, j] > 0:
                playable_actions.append(j)

        # Picking an action randomy from the list of playable actions leading
        # to the next state
        next_state = np.random.choice(playable_actions)

        # Calculating the temporal difference. The actions here refere to going
        # to the next state. The following equation is used to calculate the temporal difference
        # TD(s,a) = R(s,a) + gamma * Q(s',a') - Q(s,a)
        TD = rewards_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]

        # Updating the Q-value using the Bellman equation
        Q[current_state, next_state] += alpha*TD

    # Initializing the optimal route with the satrting location
    route = [start_location]

    # Initializing the next_location with start_location, since it is unknown
    next_location = start_location

    # Iterating till the goal location is found
    while (next_location != end_location):

        # Getting the starting state
        starting_state = location_to_state[start_location]

        # Get the highest Q-value pertaining to starting state
        next_state = np.argmax(Q[starting_state,])

        next_location = state_to_location[next_state]
        route.append(next_location)

        # Update the starting location for the next iteration
        start_location = next_location

    return route

# Taking the input from the user
start = 'L9'
end = 'L1'

print('Path for {}  to {} is: {}'.format(start, end, get_optimal_route(start, end)))
