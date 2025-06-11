# --- SwimmingThymio_Segmented.py ---

import asyncio
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tdmclient import ClientAsync
from datetime import datetime

# Global setup
client = ClientAsync()
node = None
ALPHA = 0.2
EPSILON = 0.4
motor_speed = 80
detection_threshold = 10
actions = [0, 1, 2]
states = ["R0, L0", "R0, L180", "R180, L0", "R180, L180"]
accuracies = []
Q_table = {state: {action: 0 for action in actions} for state in states}
mistake_counter = 0
time_180 = None
time_360 = None

live_info = {"state": "", "action": "", "reward": "", "next_state": ""}
testing_loop_active = False

q_table_gui_callback = lambda: None

state_labels = {
    "R0, L0": "Right Arm Front, Left Arm Front",
    "R0, L180": "Right Arm Front, Left Arm Back",
    "R180, L0": "Right Arm Back, Left Arm Front",
    "R180, L180": "Right Arm Back, Left Arm Back"
}

action_labels = {
    0: "Move Right Arm",
    1: "Move Left Arm",
    2: "Move Both Arms"
}

def set_gui_callback(callback):
    global q_table_gui_callback
    q_table_gui_callback = callback


# -----------------------------------------
# 🧠 WRAPPED FUNCTIONAL BLOCKS
# -----------------------------------------

def motors(left, right):
    """Allows motor movement. """
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

async def get_sensor_values(sensor_id, delay=0.0008):
    """Reads sensor values after a short delay. Gives exterior sensors (right and left). """
    node = await client.wait_for_node()
    await node.wait_for_variables({sensor_id})
    await client.sleep(delay)  # Allow sensor values to update
    sensor_values = list(node[sensor_id])  # Retrieve the sensor values
    l_sensor = sensor_values[1]
    r_sensor = sensor_values[3]
    bl_sensor = sensor_values[5]
    br_sensor = sensor_values[6]
    return l_sensor, r_sensor, bl_sensor, br_sensor

async def initialize_motors(motor_speed, threshold):
    """ Function to ensure both motors start at the same position """
    print("Initialization")
    # Variables
    # Read the sensor values
    l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")  # Read proximity sensors
        
    # Activate flags because there is a delay when sensed
    l_active, r_active = 0, 0

    # Until both front sensors sense something, the motors will keep running
    while not (l_active and r_active):  # Keep running until both are active
        l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")  # Read proximity sensors
        # print(f"Left: {l_sensor}, Right: {r_sensor}")

        # Check if left sensor is activated
        if l_sensor > threshold:
            l_active = 1
        # Check if right sensor is activated
        if r_sensor > threshold:
            r_active = 1

        # Control motors based on sensor states
        if l_active and r_active:
            node.send_set_variables(motors(0, 0))
            # print("Both motors stopped.")
        elif l_active:
            node.send_set_variables(motors(0, motor_speed))
            # print("Left stopped, Right moving.")
        elif r_active:
            node.send_set_variables(motors(motor_speed, 0))
            # print("Right stopped, Left moving.")
        else:
            node.send_set_variables(motors(motor_speed, motor_speed)) 
            # print("Both moving.")

    node.send_set_variables(motors(0, 0))  # Ensure final stop
    state = "R0, L0"
    return state

async def initialization_left(threshold):
    # Read the sensor values
    l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")  # Read proximity sensors
    # Activate flags because there is a delay when sensed
    l_active = 0

    # Until both front sensors sense something, the motors will keep running
    while not l_active:  # Keep running until both are active
        l_sensor, _, bl_sensor, _ = await get_sensor_values("prox.horizontal")  # Read proximity sensors
        # print(f"Left: {l_sensor}, Right: {r_sensor}")

        # Check if left sensor is activated
        if l_sensor > threshold:
            l_active = 1
            node.send_set_variables(motors(0, 0))
        else:
            node.send_set_variables(motors(motor_speed, motor_speed)) 
            # print("Both moving.")

    node.send_set_variables(motors(0, 0))  # Ensure final stop
    
async def determine_180_360(motor_speed, threshold):
    # motor_speed = abs(motor_speed)
    print("Making sure left motor starts at initial position")
    await initialization_left(detection_threshold)
    node.send_set_variables(motors(0, 0)) 
    
    # Start timer
    print("Starting timing")
    start_time = time.time()

    # Clear any lingering sensor values
    l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")
    if l_sensor > threshold:
        # Continue to move until no longer sensed, then repeat and stop until sensed
        while l_sensor > threshold or bl_sensor > threshold:
            # print("Left sensor was active at start, moving until off.")
            l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")
            node.send_set_variables(motors(motor_speed, 0))  

    l_active = False
    bl_active = False
    time_180 = 0
    time_360 = 0
    
    while not l_active: 
        node.send_set_variables(motors(motor_speed, 0)) 
        l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")
        
        # Register 360° rotation time
        if l_sensor > threshold:
            print("Front sensor active")
            l_active = True
            time_360 = time.time()
            
        # Register 180° rotation time ONCE
        if bl_sensor > threshold and not bl_active and not l_active:
            print("Backsensor active")
            bl_active = True
            time_180 = time.time()

    rotation_time = time_360 - start_time
    print(f"\n✅ 360° rotation time: {rotation_time:.2f} seconds")
    half_rotation_duration = time_180 - start_time
    print(f"\n✅ 180° rotation time: {half_rotation_duration:.2f} seconds")
    
    return rotation_time, half_rotation_duration

async def average_rotation_times(motor_speed, threshold, num_trials=3):
    rotation_times = []
    half_rotation_times = []

    for trial in range(num_trials):
        print(f"\n🔄 Running trial {trial + 1}...")
        rotation_time, half_time = await determine_180_360(motor_speed, threshold)
        # Reruns 
        if rotation_time > 10 or rotation_time < 3:
            print("⚠️ Front sensor malfunction. Retrying...")
            rotation_time, half_time = await determine_180_360(motor_speed, threshold)
        if half_time > 5 or half_time < 1:
            print("⚠️ Back sensor malfunction. Retrying...")
            rotation_time, half_time = await determine_180_360(motor_speed, threshold)
        rotation_times.append(rotation_time)
        half_rotation_times.append(half_time)

    avg_time_360 = sum(rotation_times) / num_trials
    avg_time_180 = sum(half_rotation_times) / num_trials
    print(f"\n✅ Average 180° rotation time: {avg_time_180:.2f} seconds")
    print(f"\n✅ Average 360° rotation time: {avg_time_360:.2f} seconds")
    return avg_time_360, avg_time_180

def activate_motors(action, state, time_180):
    """Execute action according to action sent."""
    # states = ["R0, L0", "R0, L180", "R180, L0", "R180, L180"]
    # print((f"Current State: {state}"))
    if action == 0:  # Turn on right motor forward (left off)
        node.send_set_variables(motors(0, motor_speed))
        state_transitions = {
            "R0, L0": "R180, L0",
            "R0, L180": "R180, L180",
            "R180, L0": "R0, L0",
            "R180, L180": "R0, L180"
        }
        next_state = state_transitions[state]
        
    elif action == 1:  # Turn on left motor forward (right off)
        node.send_set_variables(motors(motor_speed, 0))
        state_transitions = {
            "R0, L0": "R0, L180",
            "R0, L180": "R0, L0",
            "R180, L0": "R180, L180",
            "R180, L180": "R180, L0"
        }
        next_state = state_transitions[state]
    elif action == 2:  # Turn on both motors
        node.send_set_variables(motors(motor_speed, motor_speed))
        state_transitions = {
            "R0, L0": "R180, L180",
            "R0, L180": "R180, L0",
            "R180, L0": "R0, L180",
            "R180, L180": "R0, L0"
        }
        next_state = state_transitions[state]

    time.sleep(time_180) # Keep the motor on for the time it takes to do 180°, then turn off
    node.send_set_variables(motors(0, 0))    
    # print((f"Next state according to action: {next_state}"))
    return next_state

def check_correct_time(start_time, stop_time, reference_time):
    # Calculate how long it took from start to stop
    time_elapsed = stop_time - start_time
    # print(f"⏱️ Time elapsed: {time_elapsed:.2f} s (expected: {reference_time:.2f} s)")

    # Define acceptable bounds (±30%) CHANGE BOUNDS HERE, works with 30%, walking with 40%
    lower_bound = 0.60 * reference_time
    upper_bound = 1.4 * reference_time

    # Check if the elapsed time is within range
    if lower_bound <= time_elapsed <= upper_bound:
        return True
    else:
        print(f"❌ Elapsed time {time_elapsed:.2f}s is outside the allowed range ({lower_bound:.2f}s to {upper_bound:.2f}s)")
        return False

async def activate_motors_sensors(action, state, time_360, time_180, threshold):
    # Set initial state
    condition = None  
    start_time = time.time()

    # Set motor command and expected next state based on current state and action
    if action == 0:
        node.send_set_variables(motors(0, motor_speed))  # Move right motor only
        if state == "R0, L0": next_state, condition = "R180, L0", 1
        elif state == "R0, L180": next_state, condition = "R180, L180", 1
        elif state == "R180, L0": next_state, condition = "R0, L0", 2
        elif state == "R180, L180": next_state, condition = "R0, L180", 2
    elif action == 1:
        node.send_set_variables(motors(motor_speed, 0))  # Move left motor only
        if state == "R0, L0": next_state, condition = "R0, L180", 3
        elif state == "R0, L180": next_state, condition = "R0, L0", 4
        elif state == "R180, L0": next_state, condition = "R180, L180", 3
        elif state == "R180, L180": next_state, condition = "R180, L0", 4
    elif action == 2:
        node.send_set_variables(motors(motor_speed, motor_speed))  # Move both motors
        if state == "R0, L0": next_state, condition = "R180, L180", 5
        elif state == "R0, L180": next_state, condition = "R180, L0", 7
        elif state == "R180, L0": next_state, condition = "R0, L180", 8
        elif state == "R180, L180": next_state, condition = "R0, L0", 6

    # --- SINGLE MOTOR CONDITIONS ---
    print(f"Condition: {condition}")
    # Condition 1: Right motor should stop at 180° (back-right sensor)
    if condition == 1:
        print("Keep the right motor running until 360° sensor activates")
        while True:
            l, r, bl, br = await get_sensor_values("prox.horizontal")
            if br > threshold and r < threshold:
                if check_correct_time(start_time, time.time(), time_180):
                    node.send_set_variables(motors(0, 0))
                    break
                else:
                    # Wait until 360° sensor is passed, then restart timer
                    while True:
                        _, r, _, _ = await get_sensor_values("prox.horizontal")
                        if r > threshold:
                            start_time = time.time()
                            break
    # **Condition 2:** Keep the right motor running until 360° sensor activates & time is similar
    elif condition == 2:
        print("Keep the right motor running until 360° sensor activates")
        while True:
            _, r, _, _ = await get_sensor_values("prox.horizontal")
            if r > threshold:
                if check_correct_time(start_time, time.time(), time_360 - time_180):
                    node.send_set_variables(motors(0, 0))
                    break
                else:
                    while True:
                        *_, br = await get_sensor_values("prox.horizontal")
                        if br > threshold:
                            start_time = time.time()
                            break
    # **Condition 3:** Keep the left motor running until 180° sensor activates & time is similar
    elif condition == 3:
        print("Keep the left motor running until 180° sensor activates")
        while True:
            l, r, bl, br = await get_sensor_values("prox.horizontal")
            if bl > threshold and l < threshold:
                if check_correct_time(start_time, time.time(), time_180):
                    node.send_set_variables(motors(0, 0))
                    break
                else:
                    while True:
                        l, *_ = await get_sensor_values("prox.horizontal")
                        if l > threshold:
                            start_time = time.time()
                            break
    # **Condition 4:** Keep the left motor running until 360° sensor activates & time is similar
    elif condition == 4:
        print("Keep the left motor running until 360° sensor activates")
        while True:
            l, *_ = await get_sensor_values("prox.horizontal")
            if l > threshold:
                if check_correct_time(start_time, time.time(), time_360 - time_180):
                    node.send_set_variables(motors(0, 0))
                    break
                else:
                    while True:
                        *_, bl, _ = await get_sensor_values("prox.horizontal")
                        if bl > threshold:
                            start_time = time.time()
                            break

    # --- DUAL MOTOR CONDITIONS ---
    # **Condition 5:** Keep the motors running until both 180° 
    elif condition == 5:
        print("Keep the motors running until both 180°")
        r_done = False
        l_done = False

        while not (r_done and l_done):
            l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")

            if br_sensor > threshold and r_sensor < threshold and not r_done:
                node.send_set_variables(motors(motor_speed, 0))  # Stop right motor
                r_done = True

            if bl_sensor > threshold and l_sensor < threshold and not l_done:
                node.send_set_variables(motors(0, motor_speed))  # Stop left motor
                l_done = True

        node.send_set_variables(motors(0, 0))

    # **Condition 6:** Keep the motors running until both 360° 
    elif condition == 6:
        print("Keep the motors running until both 360°")
        r_done = False
        l_done = False

        while not (r_done and l_done):
            l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")

            if r_sensor > threshold and not r_done:
                node.send_set_variables(motors(motor_speed, 0))  # Stop right motor
                r_done = True

            if l_sensor > threshold and not l_done:
                node.send_set_variables(motors(0, motor_speed))  # Stop left motor
                l_done = True

        node.send_set_variables(motors(0, 0))

    # **Condition 7:** Keep the motors running until 180° right, 360° left & time is similar
    elif condition == 7:
        print("Keep the motors running until 180° right, 360° left")
        r_done = False
        l_done = False

        while not (r_done and l_done):
            l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")

            if br_sensor > threshold and r_sensor < threshold and not r_done:
                node.send_set_variables(motors(motor_speed, 0))  # Stop right motor
                r_done = True

            if l_sensor > threshold and not l_done:
                node.send_set_variables(motors(0, motor_speed))  # Stop left motor
                l_done = True

        node.send_set_variables(motors(0, 0))

    # **Condition 8:** Keep the motors running until 360° right, 180° left & time is similar
    elif condition == 8:
        print("Keep the motors running until 360° right, 180° left")
        r_done = False
        l_done = False

        while not (r_done and l_done):
            l_sensor, r_sensor, bl_sensor, br_sensor = await get_sensor_values("prox.horizontal")

            if r_sensor > threshold and not r_done:
                node.send_set_variables(motors(motor_speed, 0))  # Stop right motor
                r_done = True

            if bl_sensor > threshold and l_sensor < threshold and not l_done:
                node.send_set_variables(motors(0, motor_speed))  # Stop left motor
                l_done = True

        node.send_set_variables(motors(0, 0))
    return next_state

def get_reward(state, action):
    """ Reward function: returns +1 if the chosen action is in the list of correct actions, else -1. Behavior should be front crawl (asynch)"""
    correct_association = {
        "R0, L0": [0, 1],      # One of the motors should activate so that they are dephased, 0 turns right, 1 turns left
        "R0, L180": [2],       # They are correctly dephased so they should keep moving together
        "R180, L0": [2],       # They are correctly dephased so they should keep moving together
        "R180, L180": [0, 1]   # One of the motors should activate so that they are dephased, 0 turns right, 1 turns left
    }
    
    # If the chosen action is in the correct list, return +1; otherwise, return -1
    return 1 if action in correct_association[state] else -1

# %%
async def get_buttons(delay=0.001):
    """Reads button values after a short delay."""
    node = await client.wait_for_node()
    await node.wait_for_variables({"button.forward", "button.backward"})
    forward_button = node.var["button.forward"][0]  
    backward_button = node.var["button.backward"][0]  
    # print(f"Forward: {forward_button}, Backward: {backward_button}") 
    await client.sleep(delay)  # Allow sensor values to update
    return forward_button, backward_button

async def get_reward_child(state, action):
    """Wait for button input and return reward based on correct actions."""
    global mistake_counter

    correct_association = {
        "R0, L0": [0, 1],      # One motor should activate to dephase
        "R0, L180": [2],       # Motors are dephased, should move together
        "R180, L0": [2],       # Motors are dephased, should move together
        "R180, L180": [0, 1]   # One motor should activate to dephase
    }

    # Turn on LED before waiting for input & get actual correct reward
    if action in correct_association[state]:
        node.send_set_variables({"leds.top": [0, 32, 0]})  # Green LED ON (Correct action expected)
        correct_reward = 1
    else:
        node.send_set_variables({"leds.top": [32, 0, 0]})  # Red LED ON (Incorrect action expected)
        correct_reward = -1


    # **Wait until a button is pressed**
    while True:
        forward_button, backward_button = await get_buttons()
        if forward_button == 1:  # Forward button pressed
            reward = 1
            if reward != correct_reward :
                is_correct = False
                mistake_counter += 1
                print("❌")
            else:
                is_correct = True
                print("✅")
            break
        elif backward_button == 1:  # Backward button pressed
            reward = -1
            if reward != correct_reward :
                is_correct = False
                mistake_counter += 1
                print("❌")
            else:
                is_correct = True
                print("✅")
            break
        await client.sleep(0.2)  # Small delay to avoid busy-waiting

    # **Turn off all LEDs after button press**
    node.send_set_variables({"leds.top": [0, 0, 0], "leds.bottom.left": [0, 0, 0]})

    return reward, is_correct

def select_action(state):
    """ Epsilon-greedy policy for action selection."""
    if random.randint(0, 10) < EPSILON * 10:  # Scaled for integer compatibility
        print("I'm exploring")
        return random.choice(actions)  # Exploration action, if the agent explores, it picks a random action.
    else:
        max_value = max(Q_table[state].values())
        best_actions = [a for a in actions if Q_table[state][a] == max_value]
        return random.choice(best_actions) # Exploitation action, if it exploits, it picks the best-known action for a given state.

def update_q_table(state, action, reward, next_state, GAMMA, is_final=False):
    global live_info

    """ Q-learning update rule. """
    best_next_action = max(Q_table[next_state], key=Q_table[next_state].get)
    best_next_q = Q_table[next_state][best_next_action]
    current_q = Q_table[state][action]

    # Q-learning update
    updated_q = current_q + ALPHA * (reward + GAMMA * best_next_q - current_q)
    Q_table[state][action] = updated_q

    # Update live info
    live_info["state"] = state
    live_info["action"] = f"A{action}"
    live_info["reward"] = reward
    live_info["next_state"] = next_state

    # Debug prints
    # print(f"Best Next Action: {best_next_action}")
    print(f"State: {state} | Action: {action} | Reward: {reward} | Next State: {next_state}")
    # print(f"Q[{state}][{action}] updated from {current_q:.3f} to {updated_q:.3f}")
    print(f"  → Q(s,a) = {current_q:.3f} + ALPHA {ALPHA} * (reward {reward} + GAMMA {GAMMA} * max_next_q {best_next_q:.3f} - current_q {current_q:.3f})")
    print("Current Q-table:")
    for s in Q_table:
        print(f"  {s}: {Q_table[s]}")

    # Call GUI update callback without switching frame here
    try:
        q_table_gui_callback(is_final)
    except Exception as e:
        print("[GUI UPDATE ERROR]", e)


def get_state(initial, next_state):
    """Read motor positions to determine state. Based on activation of 1 sensor. """
    # states = ["R0, L0", "R0, L180", "R180, L0", "R180, L180"]
    if initial:
        print("I'm at the starting position")
        state = "R0, L0"
    else:
        state = next_state
    return state

def det_next_state(action, state):
    # Set motor command and expected next state based on current state and action
    if action == 0:
        if state == "R0, L0": next_state = "R180, L0"
        elif state == "R0, L180": next_state = "R180, L180"
        elif state == "R180, L0": next_state = "R0, L0"
        elif state == "R180, L180": next_state = "R0, L180"
    elif action == 1:
        if state == "R0, L0": next_state = "R0, L180"
        elif state == "R0, L180": next_state = "R0, L0"
        elif state == "R180, L0": next_state = "R180, L180"
        elif state == "R180, L180": next_state = "R180, L0"
    elif action == 2:
        if state == "R0, L0": next_state = "R180, L180"
        elif state == "R0, L180": next_state = "R180, L0"
        elif state == "R180, L0": next_state = "R0, L180"
        elif state == "R180, L180": next_state = "R0, L0"

    return next_state

# Training to get the reward when it does the correct action
def pre_training(n_episodes): 
    """Train the Q-learning agent using state-action-reward-state (SARS) updates."""
    # Go to 0,0 initially to guarantee knowing the next states
    initial = True
    next_state = "R0, L180" # Set it to something initially, can be anything
    GAMMA = 0.9
    
    # Q-learning SARS: state-action-reward-state 
    for episode in range(n_episodes):  # Number of training episodes
        # STATE
        state = get_state(initial, next_state)  # Get current state
        initial = False
        print((f"State: {state}"))
        # ACTION
        action = select_action(state)
        print((f"Action: {action}"))
        # Next state is observed after action is taken because there are no sensors, based on finite state machine
        next_state = det_next_state(action, state) # Get the next_state based on action
        print((f"Next State: {next_state}"))
        # REWARD
        reward = get_reward(state, action)  # Get reward
        # UPDATE Q-TABLE
        update_q_table(state, action, reward, next_state, GAMMA, is_final=(episode == n_episodes - 1))


        # Test accuracy every 10 episodes
        if episode % 3 == 0:
            correct_predictions = sum(
                max(Q_table[s], key=Q_table[s].get) in {"R0, L0": [0,1], "R0, L180": [2], "R180, L0": [2], "R180, L180": [0,1]}[s]
                for s in states
            )
            accuracy = (correct_predictions / len(states)) * 100
            accuracies.append((episode, accuracy))
            print(f"Episode {episode}: Accuracy = {accuracy:.2f}%")

# Training to get the reward when it does the correct action
async def training(n_episodes, time_360, time_180, threshold, level): 
    """Train the Q-learning agent using state-action-reward-state (SARS) updates."""
    # Go to 0,0 initially to guarantee knowing the next states
    with open("Thymio_Episode_QTables.txt", "w") as f:
        f.write("=== Thymio Q-Learning Training Log ===\n\n")

    state = await initialize_motors(motor_speed, threshold)
    initial = True
    next_state = "R0, L180" # Set it to something initially, can be anything

    training_logs = [] 
    
    # Q-learning SARS: state-action-reward-state 
    for episode in range(n_episodes):  # Number of training episodes
        # STATE
        state = get_state(initial, next_state)  # Get current state
        initial = False
        # print((f"State: {state}"))
        
        # ACTION
        action = select_action(state)  # Choose an action based on greedy based policy
        # print(f"Action: {action}")
        
        # Next state is observed after action is taken because there are no sensors, based on finite state machine
        next_state = await activate_motors_sensors(action, state, time_360, time_180, detection_threshold) # Do the action associated with that state
        # print(f"Next State: {next_state}")
        
        # REWARD
        reward, is_correct = await get_reward_child(state, action)

        # Discount Factor changes depending on moment in training (Further training carries more weight)
        if level == 2:
            GAMMA = 0.9  # Fixed gamma for Activity 2
        else:
            gamma_correct = 0.9
            gamma_wrong_start = 0.5
            gamma_wrong = gamma_wrong_start + (gamma_correct - gamma_wrong_start) * (episode / (n_episodes - 1))
            GAMMA = gamma_correct if is_correct else gamma_wrong

        print(f"Gamma: {GAMMA}")
        # print(f"Reward: {reward}")
        
        update_q_table(state, action, reward, next_state, GAMMA, is_final=(episode == n_episodes - 1))
        training_logs.append({
            "episode": episode + 1,
            "initial_state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })

        save_q_table_snapshot(episode + 1)

        # Test accuracy every 10 episodes
        if episode % 3 == 0:
            correct_predictions = sum(
                max(Q_table[s], key=Q_table[s].get) in {"R0, L0": [0,1], "R0, L180": [2], "R180, L0": [2], "R180, L180": [0,1]}[s]
                for s in states
            )
            accuracy = (correct_predictions / len(states)) * 100
            accuracies.append((episode, accuracy))
            print(f"Episode {episode}: Accuracy = {accuracy:.2f}%")

def plot_training_accuracy(): 
    episodes, accuracy_values = zip(*accuracies)
    
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, accuracy_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Training Episodes")
    plt.ylabel("Accuracy (%)")
    plt.title("Q-learning Accuracy Improvement Over Training")
    plt.ylim(0, 110)  # Accuracy should be between 0 and 100%
    plt.grid(True)
    plt.show()

async def testing_over_time(time_360, time_180, num_movements=10):
    """ Tests the learned policy over an extended period (at least 10 movements). """
    # Start from a known initial state
    state = await initialize_motors(motor_speed, detection_threshold)  
    print(f"\n🔄 Starting at Initial State: {state}")
    
    correct_predictions = 0
    
    for move in range(num_movements):  # Loop through at least 20 movements
        action = select_action(state)  # Choose action based on learned policy
        
        # Expected correct actions
        correct_actions = {"R0, L0": [0, 1], "R0, L180": [2], "R180, L0": [2], "R180, L180": [0, 1]}
        correct_action = correct_actions[state]  # Get expected action(s)

        # Ensure correct_action is always a list
        if not isinstance(correct_action, list):
            correct_action = [correct_action]

        # Execute action and determine new state
        next_state = await activate_motors_sensors(action, state, time_360, time_180, detection_threshold)  
        print(f"🔁 Movement {move+1}: {state} → {next_state} via Action {action}")

        # Check if the action was correct
        if action in correct_action:
            correct_predictions += 1

        # Update state for the next movement
        state = next_state

    # Compute and print accuracy over extended movements
    accuracy = (correct_predictions / num_movements) * 100
    print(f"\n✅ Final Accuracy Over {num_movements} Movements: {correct_predictions}/{num_movements} correct ({accuracy:.2f}%)")
    print("Setup: definitions and helpers ready.")

async def run_get_started():
    global Q_table, accuracies, time_180, time_360, node
    Q_table = {state: {action: 0 for action in actions} for state in states}
    accuracies.clear()
    node = await client.wait_for_node()
    await node.lock()

    time_360, time_180 = await average_rotation_times(motor_speed, detection_threshold)
    await initialize_motors(motor_speed, detection_threshold)
    node.send_set_variables(motors(0, 0))
    print(f"Timing complete. 180° = {time_180:.2f}s, 360° = {time_360:.2f}s")

def restart_learning():
    global Q_table, accuracies
    Q_table = {state: {action: 0 for action in actions} for state in states}
    accuracies.clear()
    print("Restarted learning: Q-table and accuracy log reset.")

def run_pretrain(pre_training_iter=10):
    pre_training(pre_training_iter)
    # plot_training_accuracy()
    print("Pretraining and plot complete.")

async def run_train(training_iter=15, switch_to_final=False, level=1):
    global EPSILON, mistake_counter, node
    EPSILON = 0.8
    accuracies.clear()
    mistake_counter = 0
    await training(training_iter, time_360, time_180, detection_threshold, level)
    print(f"Training complete. Total mistakes: {mistake_counter}")
    # plot_training_accuracy()
    
    # Trigger final GUI update with switch signal
    q_table_gui_callback(is_final=switch_to_final)


async def run_test(n_movements=10):
    global EPSILON, node
    EPSILON = 0.0
    await testing_over_time(time_360, time_180, num_movements=n_movements)
    print("Testing complete.")

async def disconnect_node():
    global node
    if node:
        await node.unlock()
        print("Thymio node disconnected.")

def get_q_table():
    return Q_table

def get_live_info():
    return live_info

def save_q_table_snapshot(episode_num):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("Thymio_Episode_QTables.txt", "a") as f:
        f.write(f"\n=== Episode {episode_num} @ {timestamp} ===\n")
        f.write(f"Initial State: {state_labels.get(live_info['state'], live_info['state'])}\n")
        action_num = int(live_info['action'][1]) if live_info['action'].startswith("A") else live_info['action']
        f.write(f"Action: {action_labels.get(action_num, live_info['action'])}\n")
        f.write(f"Reward: {live_info['reward']}\n")
        f.write(f"Future State: {state_labels.get(live_info['next_state'], live_info['next_state'])}\n")
        f.write("Q-Table:\n")
        for state in Q_table:
            f.write(f"  {state_labels.get(state, state)}:\n")
            for action in Q_table[state]:
                f.write(f"    {action_labels.get(action, f'A{action}')}: {Q_table[state][action]:.2f}\n")

testing_loop_active = False

def start_testing_loop_level1(max_movements=100):
    global testing_loop_active, EPSILON
    EPSILON = 0.0  # 💡 Ensure policy is greedy during testing

    async def loop():
        global testing_loop_active
        testing_loop_active = True

        # Initialize state just once
        state = await initialize_motors(motor_speed, detection_threshold)

        for move in range(max_movements):
            if not testing_loop_active:
                print("🔴 Testing manually stopped.")
                break

            action = select_action(state)
            next_state = await activate_motors_sensors(action, state, time_360, time_180, detection_threshold)

            # Optional: check correctness here if you want to log accuracy
            print(f"Movement {move+1}: {state} → {next_state} via action {action}")
            state = next_state  # Update for next movement

        print("✅ Testing completed or manually stopped.")

    import threading
    import asyncio
    threading.Thread(target=lambda: asyncio.run(loop())).start()

def stop_testing_loop_level1():
    global testing_loop_active
    testing_loop_active = False

# --- LEVEL 2 Step-by-step functions ---

async def run_train_step_level2():
    """Run one action-execute-display step for Level 2."""
    global time_360, time_180, detection_threshold, node

    if not hasattr(run_train_step_level2, "state"):
        run_train_step_level2.state = await initialize_motors(motor_speed, detection_threshold)
        run_train_step_level2.initial = False
        run_train_step_level2.episode = 0

    state = get_state(run_train_step_level2.initial, run_train_step_level2.state)
    run_train_step_level2.initial = False

    action = select_action(state)
    next_state = await activate_motors_sensors(action, state, time_360, time_180, detection_threshold)

    correct_reward = get_reward(state, action)

    live_info["state"] = state
    live_info["action"] = f"A{action}"
    live_info["reward"] = f"{correct_reward} (expected)"
    live_info["next_state"] = next_state

    try:
        q_table_gui_callback(is_final=False)
    except Exception as e:
        print("[GUI CALLBACK ERROR]", e)

    run_train_step_level2.state = next_state
    run_train_step_level2.last_state = state
    run_train_step_level2.last_action = action

async def confirm_reward_and_update_level2(level=2):
    if not hasattr(run_train_step_level2, "last_state") or not hasattr(run_train_step_level2, "state"):
        print("[ERROR] run_train_step_level2 must be called before confirming reward.")
        return

    state = run_train_step_level2.last_state
    action = run_train_step_level2.last_action
    next_state = run_train_step_level2.state

    if level == 2:
        # ✅ Use expected correct reward only; skip physical buttons
        reward = get_reward(state, action)
        GAMMA = 0.9
    else:
        # Still show LEDs and wait for student input (Level 1 only)
        reward, is_correct = await get_reward_child(state, action)
        gamma_correct = 0.9
        gamma_wrong_start = 0.5
        gamma_wrong = gamma_wrong_start + (gamma_correct - gamma_wrong_start) * (run_train_step_level2.episode / 15)
        GAMMA = gamma_correct if is_correct else gamma_wrong


    update_q_table(state, action, reward, next_state, GAMMA)
    save_q_table_snapshot(run_train_step_level2.episode + 1)
    run_train_step_level2.episode += 1