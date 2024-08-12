import numpy as np
from math import comb
from collections import defaultdict
from copy import deepcopy 

def init_draw():
    idx_dict = {}
    starting_hands = []
    for i in range (17):
        starting_hands.append((i + 4, 0))
        idx_dict[(i + 4, 0)] = i
    for i in range (17, 26):
        starting_hands.append((i - 15, 1))
        idx_dict[(i - 15, 1)] = i
    idx_dict[(0, 2)] = 26
    starting_hands.append((0, 2))
    #
    a, b = 4**2, comb(4, 2) 
    hand_prob = np.array([0, a, a, 2*a, 2*a, 3*a, 3*a, 4*a, 3*a + 64, 3*a + 64, 2*a + 64, 2*a + 64, a + 64, a + 64, 64, 64, comb(16,2), a, a, a, a, a, a, a, a, 64, b])
    for i in range (0, 16, 2):
        hand_prob[i] += b 
    omega = comb(52, 2)
    k = 1/omega 
    #
    hand_prob = np.around(np.multiply(hand_prob,k),5)
    idx_dealer = {}
    for i in range (2, 11):
        idx_dealer[(i - 2, 0)] = i - 2
    idx_dealer[(0,1)] = 9
    dealer_prob = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4, 1])
    dealer_prob = np.multiply(dealer_prob, 1/13)
    return starting_hands, hand_prob, dealer_prob, idx_dict, idx_dealer



def draw(starting_hands, hand_prob, dealer_prob):
    indices = [i for i in range (27)]
    hand = np.random.choice(indices, 1, hand_prob, replace = False)
    indices = [i for i in range (10)]
    dealer_hand = np.random.choice(indices, 1, dealer_prob, replace = False)
    hand = starting_hands[hand]
    if dealer_hand == 9:
        dealer_hand = (0, 1)
    else:
        dealer_hand = (dealer_hand + 2, 0)
    return hand, dealer_hand

def init_states():
    #s = (player_hard_score, players_aces, dealer_card)
    S = {}
    cnt = 0
    states = []
    for i in range (22):
        for j in range (22 - i): 
            for k in range (2,12):
                S[(i, j, k)] = cnt
                states.append((i,j,k))
                cnt += 1
    return S,cnt,states

def inspect_state(S, dynamics, i, j, k, mid_states = True): #stats about specific state
    flag = True
    print("STAND:", dynamics.get((S[(i,j,k)],"stand")))
    print("DOUBLE DOWN:", dynamics.get((S[(i,j,k)],"double down")))
    if mid_states:
        print("HIT:", dynamics.get((S[(i,j,k)],"hit")))
        return
    print("HIT:",end = ' ')
    for el in dynamics[(S[(i,j,k)],"hit")]:
        if type(el[0]) == str:
            if flag:
                print(el, end = '')
                flag = False
            else:
                print(' ,',el)

def init_env_dynamics(bid):
    def init_dealer_bust():
        def soft_to_hard(soft, hard): #podlicza przejscia z softa do harda
            for i in range (12, 17):
                soft[i] += 4/13*hard[i] #bo +10 i -10 bo as traci na wartosci 10
                for k in range (i - 12):
                    soft[i] += 1/13*hard[i - 1 - k]
            
        dealer_bust = np.zeros(12)
        hard, soft, directly = np.zeros(17), np.zeros(17), np.zeros(17) #directly dotyczy tylko hard, bo nie da sie z softa < 17 zbustować
        directly = [0]*(17)
        directly[16],directly[15], directly[14], directly[13], directly[12] = 8/13, 7/13, 6/13, 5/13, 4/13
        #part 1: hard 6 - 16
        hard[16] = directly[16]
        hard[15] = directly[15] + 1/13*hard[16]
        for i in range (14, 6, -1):
            if i >= 11:
                hard[i] = (sum(hard[i + 1: 17]))/13 + directly[i]
            else:
                hard[i] = (sum(hard[i + 2: 17]))/13 + directly[i]
        hard[6] = sum(hard[8: 16])/13 + 4/13*hard[16] #kwestia innego prawdopodobienstwa na dropniecie +10
        #part 2: soft 11 - 16
        soft_to_hard(soft, hard)
        for i in range (15, 10, -1):
            soft[i] += sum(soft[i + 1:17])/13
        #part 3: hard 2 - 5
        for i in range (5, 1, -1):
            hard[i] = (sum(hard[i + 2: i + 10]) + 4*hard[i + 10] + soft[i + 11])/13
        dealer_bust[:11] = hard[:11] #[0,1,2,3,4,5,6,7,8,9,10,A]
        dealer_bust[11] = soft[11]
        print(np.around(np.array(dealer_bust), 5))
        return np.around(np.array(dealer_bust), 5)

    def init_dealer_stands_at(a): #oblicza prawdopodobienstwa ze krupier zostanie na a (a>=17) w zaleznosci od k
        def soft_to_hard(soft, hard): 
            for i in range (12, 17):
                soft[i] += 4/13*hard[i]
                for k in range (i - 12):
                    soft[i] += 1/13*hard[i - 1 - k]

        directly = [0]*(17)
        for i in range (16, 5, -1):
            if i == a - 10:
                directly[i] = 4/13
            elif i == a - 11 or i > a - 10:
                directly[i] = 1/13
    
        goal_prob = np.zeros(17)
        hard = [0]*17
        soft = [0]*17
        #part 1: hard 6 - 16
        hard[16] = directly[16]
        hard[15] = directly[15] + (1/13)*hard[16] 
        for i in range (14, 6, -1):
            if i >= 11: 
                hard[i] = 1/13*sum(hard[i + 1:17]) + directly[i]
            else:
                hard[i] = (sum(hard[i + 2: 17]))/13 + directly[i]
        hard[6] = sum(hard[8: 16])/13 + 4/13*hard[16] + directly[6]
        #part 2: soft 11 - 16
        soft_to_hard(soft, hard)
        soft[16] += directly[16]
        for i in range (15, 10, -1):
            soft[i] += sum(soft[i + 1:17])/13 + directly[i]
        #part 3: hard 2 - 5
        for i in range (5, 1, -1):
            hard[i] = (sum(hard[i + 2: i + 10]) + 4*hard[i + 10] + soft[i + 11])/13
        goal_prob[:11] = hard[:11]
        goal_prob[11] = soft[11]
        return np.around(np.array(goal_prob), 5)

    def score(i,j):
        aces = j
        sc = i + 11*aces
        while aces > 0 and sc > 21:
            aces -= 1
            sc -= 10
        return sc

    def bust_chances(i,j,k):
        nonlocal dynamics, bid
        if i + j >= 12:
            dynamics[(S[(i,j,k)], "hit")].append(("BUST", -bid, round((i + j - 8)/13,5)))
            dynamics[(S[(i,j,k)], "double down")].append(("BUST", -2*bid, round((i + j - 8)/13,5)))

    def count_preSum(dealer_stand):
        probSum = deepcopy(dealer_stand)
        for k in range (2, 12):
            for i in range (1, len(probSum)):
                probSum[i][k] += probSum[i - 1][k]
        return probSum
        
    def win_chances(i,j,k,dealer_bust,dealer_stand): #wygrać można na trzy sposoby: BUST krupiera albo lepszy wynik agenta albo BJ agenta
        nonlocal dynamics, bid, probSum
        prob = dealer_bust[k] #BUST krupiera
        #oblicz score
        a,b = round(1/13,5), round(4/13,5)
        res = score(i,j)
        if res > 21: #bo to jest bust
            return
        #
        if res == 21:
            dynamics[(S[(i,j,k)], "stand")].append(("WIN", 3/2*bid, 1))
        if res <= 17:
            dynamics[(S[(i,j,k)], "stand")].append(("WIN", bid, prob))
        else:
            idx = res - 18
            dynamics[(S[(i,j,k)], "stand")].append(("WIN", bid, prob + probSum[idx][k])) #albo bust albo wynik lepszy
        # mozemy jeszcze wygrac bezposrednio poprzez BJ: czyli po HIT/DOUBLE DOWN
        if res < 10: #nie wygramy bezposrednio z tego
            return
        if res == 11:
            dynamics[(S[(i,j,k)], "hit")].append(("WIN", bid, b))
        else:
            dynamics[(S[(i,j,k)], "hit")].append(("WIN", bid, a))
    
    def push_chances(i,j,k, dealer_stand):
        #20-20,19-19,18-18,17-17
        nonlocal probSum, dynamics
        res = score(i,j)
        a = 0
        if res >= 21 or res < 6:
            return
        if 17 <= res + 10 <= 20:
            a = 3*dealer_stand[res - 7][k]
        idx = min(3, res - 6)
        prob = 1/13*(probSum[idx][k] + a)
        # print(round(prob,3), res, k)
        dynamics[(S[(i,j,k)], "double down")].append(("PUSH", 0, round(prob,5))) #remis gdy double downujemy (trzeba jeden krok do przodu prawdopodbienstwo policzyc)
        if res < 17:
            return
        dynamics[(S[(i,j,k)], "stand")].append(("PUSH", 0, round(dealer_stand[res - 17][k],5))) #remis gdy standujemy
        
    def lose_chances(i,j,k): #albo agent BUST albo mniej punktow od krupiera, ale BUST mamy jako oddzielny stan terminalny, więc tylko mniej pkt od krupiera
        nonlocal dynamics, probSum_reverse
        if score(i,j) < 17:
            dynamics[(S[(i,j,k)], "stand")].append(("LOSE", -bid, round(1 - dealer_bust[k],5))) #jesli mamy mniej niz 17 to pzregramy, jesli dealer nie zbustuje
        if 17 <= score(i,j) < 21:
            dynamics[(S[(i,j,k)], "stand")].append(("LOSE", -bid, round(probSum_reverse[-2 - (score(i,j) - 17)][k],5))) #np.dla 17 diler moze miec 18,19,20,21 

    def mid_states(i,j,k):
        nonlocal dynamics
        a,b = round(1/13,5), round(4/13,5)
        if score(i, j + 1) < 21: #gdy =21 to juz mam wziete pod uwage jako WIN poprzez BJ
            dynamics[(S[(i,j,k)], "hit")].append(((i,j + 1,k), 0, a))
        if score(i + 10,j) < 21: 
            dynamics[(S[(i,j,k)], "hit")].append(((i + 10,j,k), 0, b))
        for add in range (2,10):
            if score(i + add, j) < 21:
                dynamics[(S[(i,j,k)], "hit")].append(((i + add, j, k), 0, a))

    def complete_double_down(i,j,k):
        #LOSE
        prob = 0
        if score(i,j + 1) < 21:
            prob += 1/13*next(filter(lambda x: x[0] == "LOSE", dynamics[(S[(i,j + 1,k)], "stand")]))[2]
        for add in range (2, 10):
            if score(i + add, j) >= 21:
                break
            prob += 1/13*next(filter(lambda x: x[0] == "LOSE", dynamics[(S[(i + add,j,k)], "stand")]))[2]
        if score(i + 10,j) < 21:
            prob += 4/13*next(filter(lambda x: x[0] == "LOSE", dynamics[(S[(i + 10,j,k)], "stand")]))[2]
        dynamics[(S[(i,j,k)], "double down")].append(("LOSE", -2*bid, round(prob,5)))
        #WIN
        prob = 0
        if score(i,j + 1) < 21:
            prob += 1/13*next(filter(lambda x: x[0] == "WIN", dynamics[(S[(i,j + 1,k)], "stand")]))[2]
        elif score(i,j + 1) == 21:
            prob += 1/13
        for add in range (2, 10):
            if score(i + add, j) >= 21:
                break
            prob += 1/13*next(filter(lambda x: x[0] == "WIN", dynamics[(S[(i + add,j,k)], "stand")]))[2]
        if score(i + 10,j) < 21:
            prob += 4/13*next(filter(lambda x: x[0] == "WIN", dynamics[(S[(i + 10,j,k)], "stand")]))[2]
        elif score(i + 10,j) == 21:
            prob += 4/13
        dynamics[(S[(i,j,k)], "double down")].append(("WIN", 2*bid, round(prob,5)))
    
    A = ["hit", "stand", "double down"]
    S,cnt,states = init_states()
    dynamics = defaultdict(list) #dict[(s,a)] = [(next_s, r, probability), ...]
    dealer_bust = init_dealer_bust()
    #dealer_bust = [0.0, 0.0, 0.35, 0.37, 0.4, 0.42, 0.42, 0.26, 0.24, 0.23, 0.23, 0.17]
    dealer_stand = np.around(np.array([init_dealer_stands_at(17), init_dealer_stands_at(18), init_dealer_stands_at(19), init_dealer_stands_at(20), init_dealer_stands_at(21)]),5)
    probSum = np.around(count_preSum(dealer_stand),5)
    probSum_reverse = np.around(count_preSum(dealer_stand[::-1]),5)
    print(probSum[-1])
    for i in range (22):
        for j in range (22 - i):
            for k in range (2, 12): 
                bust_chances(i,j,k)
                win_chances(i,j,k, dealer_bust, dealer_stand)
                lose_chances(i,j,k)
                push_chances(i,j,k, dealer_stand)
                mid_states(i,j,k)
   
    for i in range (22):
        for j in range (22 - i):
            for k in range (2, 12):
                complete_double_down(i,j,k)
    return dynamics




dynamics = init_env_dynamics(1)





