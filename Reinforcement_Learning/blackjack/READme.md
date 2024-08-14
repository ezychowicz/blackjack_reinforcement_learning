# Blackjack
This is a little simplified BJ: no surrendering and no splitting.
This folder contains 2 methods for finding optimal strategy in BJ. I treat the game as basic MDP, so strategy doesn't consider card counting. The methods are:
- Dynamic Programming (Value Iteration in DP_method.py and computing environment dynamics in generate_dynamics.py)
- Exploring Starts Monte Carlo (play games by randomly choosing starting state and action (S_0, A_0) (ES), then it's normal MC procedure of policy evaluation and iteration)

Note: comparing those strategies to those I found online: there are some differences. In DP method generating environment dynamics was very tedious and sometimes complex. Some minor mistakes are possible. In MC, when differences in Q(s,a) are
small, it might take the suboptimal action.
