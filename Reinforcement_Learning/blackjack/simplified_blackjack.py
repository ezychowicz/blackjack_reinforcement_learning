import generate as gen
import numpy as np

S, states_number, states = gen.init_states()
dynamics = gen.init_env_dynamics(1)
starting_hands, hand_prob, dealer_prob, idx_hand, idx_dealer = gen.init_draw()
S["WIN"], S["LOSE"], S["PUSH"], S["BUST"] = states_number, states_number + 1, states_number + 2, states_number + 3 
states_number += 4 #stany terminalne: WIN,LOSE,BUST,PUSH (w states ich nie ma, bo to nawet nie są terminalne stany tylko oznaczenia końca bardziej)

def value_iteration(S, states_number, dynamics, epsilon = 0.0000001, discount = 1): #daje polityke deterministyczną (nie stochastyczną)
    A = ["hit", "stand", "double down"]
    def extract_policy():
        nonlocal S, dynamics, v
        policy = [0]*states_number
        for s in states:
            q = np.zeros(3) #troche inny zapis niz ponizej bo chcialem zapisac za pomocą q-value tutaj
            for i,a in enumerate(A):
                suma = 0
                possibilities = dynamics[(S[s], a)]
                for el in possibilities:
                    next_s, reward, trans_prob = el
                    suma += trans_prob*(reward + discount*v[S[next_s]])
                q[i] = suma
            policy[S[s]] = A[np.argmax(q)]
        return policy
    
    v = np.zeros(states_number)
    max_diff = 0 #delta
    while max_diff < epsilon:
        max_diff = 0
        for s in states: 
            old = v[S[s]]
            max_sum = -float('inf')
            for a in A:
                suma = 0
                possibilities = dynamics[(S[s], a)]
                for el in possibilities:
                    next_s, reward, trans_prob = el
                    suma += trans_prob*(reward + discount*v[S[next_s]])
                max_sum = max(max_sum, suma)
            v[S[s]] = max_sum
            max_diff = max(max_diff, abs(v[S[s]] - old))

    return extract_policy(), v

policy, v = value_iteration(S, states_number, dynamics)

for hand in starting_hands:
    print("our hand:", hand)
    for k in range (2, 12):
        print("for dealer's", k, policy[S[(hand[0], hand[1], k)]])
    print("_____________")

np.set_printoptions(threshold=np.inf)

def calculate_q(S,s,a,discount = 1):
    possibilities = dynamics[(S[s], a)]
    suma = 0
    for el in possibilities:
        next_s, reward, trans_prob = el
        suma += trans_prob*(reward + discount*v[S[next_s]])
    return suma

def inspect_value(S, dynamics, i, j, k, mid_states = True):
    gen.inspect_state(S,dynamics,i,j,k)
    for a in ["stand", "hit", "double down"]:
        print("E of action", a, calculate_q(S,(i,j,k),a))

inspect_value(S,dynamics, 5,1,6)