import numpy as np
import matplotlib.pyplot as plt

class KArmedBandit:
    def __init__(self, k, epsilon, alpha, sd):
        self.k = k
        self.epsilon = epsilon
        self.q_val = np.full(k, 0)
        #self.q_val =  np.random.normal(0, 1, k)
        self.q_estimates_alpha = np.zeros(k)
        self.q_estimates_sa = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.alpha = alpha
        self.sd = sd
        
    def select_action(self):
        self.val+= np.random.normal(0, self.sd, self.k)
        reward_sa=self.take_action(self.q_estimates_sa, "sa")
        reward_alpha=self.take_action(self.q_estimates_alpha, "alpha")

        return reward_sa, reward_alpha

    def take_action(self, q_estimates, type):
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.k)
        else:
            action = np.argmax(self.q_estimates)

        reward = np.random.normal(self.q_val[action], 1)
        if(type=="sa"):
            self.update_estimates_sa(action, reward)
        else:
            self.update_estimates_alpha(action, reward)
    
        return reward

    def update_estimates_sa(self, action, reward):
        self.action_counts[action] += 1
        self.q_estimates_sa[action] =self.q_estimates_sa[action] + (reward - self.q_estimates_sa[action]) / self.action_counts[action]

    def update_estimates_alpha(self, action, reward):
        self.action_counts[action] += 1
        self.q_estimates_alpha[action] =self.q_estimates_alpha[action] + self.alpha*(reward - self.q_estimates_sa[action])
def run_bandit_experiment(k, epsilon, steps, runs, alpha, sd):
    rewards_history_sa = np.zeros((runs, steps))
    rewards_history_alpha = np.zeros((runs, steps))

    for run in range(runs):
        bandit = KArmedBandit(k, epsilon, alpha, sd)
        for step in range(steps):
            reward_sa, reward_alpha = bandit.choose_action()
            rewards_history_sa[run, step] = reward_sa
            rewards_history_alpha[run, step] = reward_alpha

    average_rewards_sa = np.mean(rewards_history_sa, axis=0)
    average_rewards_alpha = np.mean(rewards_history_alpha, axis=0)

    return average_rewards_sa, average_rewards_alpha

def main():
    k = 10
    steps = 10
    runs = 2000
    epsilon = 0.1
    alpha = 0.1
    sd=0.01

    average_rewards_sa, average_rewards_alpha = run_bandit_experiment(k, epsilon, steps, runs, alpha, sd)

    plt.plot(average_rewards_sa, label='Sample Average')
    plt.plot(average_rewards_alpha, label='Constant Step-Size (alpha=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('K-Armed Bandit - Epsilon-Greedy')
    plt.legend()
    plt.show()

main()

##pabaigt vizualizacija