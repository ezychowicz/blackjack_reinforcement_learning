#RL using Exploring Start policy Monte Carlo method
import random
from simplified_blackjack import starting_hands
import numpy as np
from collections import defaultdict
def init_states():  #have to modify, you can only double down at the beginning. In DP method I managed to avoid adding a dimension to a state by computing additional probabilities (one step forward for after double down)
    #s = (player_hard_score, players_aces, dealer_card, is_start)
    S = {}
    cnt = 0
    states = []
    for i in range (22):
        for j in range (22 - i): 
            for k in range (1,11):
                for is_start in [True, False]:
                    S[(i, j, k, is_start)] = cnt
                    states.append((i,j,k))
                    cnt += 1
    return S,cnt,states

S, states_number, states = init_states() #S[(i,j,k)] = idx

def score(pair):
    i,j = pair 
    aces = j
    sc = i + 11*aces
    while aces > 0 and sc > 21:
        aces -= 1
        sc -= 10
    return sc

def ES_montecarlo(S, states_number, games = 10000000):
    def update(hand, card):
        if card == 1:
            hand = (hand[0], hand[1] + 1)
        else:
            hand = (hand[0] + card, hand[1])
        return hand
    def calculate_return(path, dealer_hand, double_down):
        if len(path) == 1 and score((path[0][0][0], path[0][0][1])) == 21:
            return 3/2
        #simulate dealer's result:
        k = double_down + 1
        while score(dealer_hand) < 17:
            add = random.choice(cards)
            dealer_hand = update(dealer_hand, add)
        dealer_score = score(dealer_hand)
        agent_score = score((path[-1][0][0],path[-1][0][1]))
        if agent_score == 21:
            return k
        if agent_score > 21:
            return -k
        if dealer_score > 21:
            return k
        if agent_score < dealer_score:
            return -k
        if agent_score == dealer_score:
            return 0
        return k
    def generate_episode(state, action):
        nonlocal path
        if action == "hit" or action == "double down":
            card = random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])
            if card == 1:
                next_state = (state[0], state[1] + 1, state[2], False)
            else:
                next_state = (state[0] + card, state[1], state[2], False) 
            if score((next_state[0], next_state[1])) > 21:
                path.append((next_state, None))
                return
            action = policy[S[next_state]]
            path.append((next_state, action))
            
            generate_episode(next_state, action)

    cards = [1,2,3,4,5,6,7,8,9,10,10,10,10] #A,2,3,4,...,J,Q,K
    A = ["hit", "stand", "double down"]
    # cnts = [[0, 0, 0] for _ in range(states_number)]
    cnts = [[0, 0, 0] for _ in range(states_number)]
    action_dict = {}
    action_dict["hit"], action_dict["stand"], action_dict["double down"] = 0, 1, 2
    def choose(cards):
        card1, card2 = random.choice(cards), random.choice(cards)
        dealer_card = random.choice(cards)
        aces = 0
        if card1 == 1:
            aces += 1
            card1 = 0
        if card2 == 1:
            aces += 1
            card2 = 0
        hand = (card1 + card2, aces)
        return hand[0], hand[1], dealer_card
    
    policy = ["hit" for _ in range (states_number)] #random.choice(A)
    q = [np.zeros(3) for _ in range(states_number)] #q value for every s,a pair
    for i in range (22):
        for j in range (22 - i): 
            for k in range (1,11):
                q[S[(i,j,k,False)]] = np.zeros(2)


    game = 0
    while game < games:
        i,j,k = choose(cards) #random choice of starting state
        dealer_hand = update((0,0),k)
        action = random.choice(A)
        path = [((i, j, k, True),action)]

        double_down = False
        if action == "double down":
            double_down = True

        generate_episode((i,j,k,True),action)
        ret = calculate_return(path, dealer_hand, double_down) #no loop needed, return will be one and universal for all pairs s,a because: no discounting, return = terminal reward
        #print(path, ret)
        for s, a in path:
            if s not in S: #terminal state
                continue
            cnt = cnts[S[s]][action_dict[a]]
            old = q[S[s]][action_dict[a]]
            q[S[s]][action_dict[a]] = (cnt*q[S[s]][action_dict[a]] + ret)/(cnt + 1)  
            #q[S[s]][action_dict[a]] = q[S[s]][action_dict[a]] + 1/cnt*(ret - q[S[s]][action_dict[a]]) #cnts=[(1,1,1)...]
            cnts[S[s]][action_dict[a]] += 1
        for s,a in path:
            if s not in S: 
                continue
            policy[S[s]] = A[np.argmax(q[S[s]])]
        game += 1

    return policy, q
    


import time
if __name__ == "__main__":
    A = ["hit", "stand", "double down"]
    sums = defaultdict(lambda: np.zeros(3))
    n = 100
    for samples in range (n):  #average of n policies
        policy, q = ES_montecarlo(S,states_number)
        print(time.time())
        for hand in starting_hands:
            for k in range (1, 11):
                sums[(hand,k)] = sums[(hand,k)] + q[S[(hand[0],hand[1], k, True)]]
                #print(sums[(hand,k)])

    for hand in starting_hands:
        print("our hand:", hand)
        for k in range (2, 11):
            #print(np.multiply(sums[(hand,k)],1/n))
            print(f"for dealer's: {k} {A[np.argmax(sums[(hand,k)])]}")
        #print(f"hit, stand, double down: {np.multiply(sums[(hand,1)],1/n)}")
        print(f"for dealer's: A {A[np.argmax(sums[(hand,1)])]}")
        print("_____________")