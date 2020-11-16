import time
import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

class Bandit(object):
    def __init__(self,n,thetas):
        assert len(thetas) == n
        self.n = n
        self.thetas = thetas
        self.best_theta = max(self.thetas)

    def pull_arm(self, i):
        # Pull i-th arm and return reward
        if np.random.random() < self.thetas[i]:
            return 1
        else:
            return 0

# Create a bandit instance

thetas = [0.15,0.30,0.60]
bandit = Bandit(3,thetas) #Define a 3-armed bandit

# Define a generic solver
class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve
        """
        assert isinstance (bandit, Bandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = [] # History of actions (list of arms pulled)
        self.regret = 0. # Cumulative regret
        self.regrets = [0.] #History of cumulative regrets

    def update_regret(self, i):
        # i: index of selected arm
        self.regret += self.bandit.best_theta - self.bandit.thetas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_thetas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the arm index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


# Implement some standard solvers

# 1. Random (explore-only)
class Random(Solver):
    def __init__(self, bandit):
        super(Random, self).__init__(bandit)
        self.name = "Random"

    def run_one_step(self):
            i = np.random.randint(0, self.bandit.n) #Pick a random arm
            reward = self.bandit.pull_arm(i) #Pull arm i and get reward
            return i

# 2. Greedy (exploit-only)
class Greedy(Solver):
    def __init__(self, bandit, arm):
        super(Greedy, self).__init__(bandit)
        self.arm = arm # Chosen arm ("best-known")
        self.name = "Greedy"

    def run_one_step(self):
        reward = self.bandit.pull_arm(self.arm) #Pull arm i and get reward
        return self.arm

#3. epsilon-greedy
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon, init_theta=1.0):
        """
        epsilon (float): the probability to explore
        init_theta (float): We optimistically set initial theta to be 1.0
        """
        super(EpsilonGreedy, self).__init__(bandit)
        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon
        self.estimates = [init_theta]*self.bandit.n #Optimisistic initialisation
        self.name = "Epsilon-Greedy"
    @property
    def estimated_thetas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            #We are going to explore
            i = np.random.randint(0, self.bandit.n) #Pick a random arm
        else:
            #We are going to exploit
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        reward = self.bandit.pull_arm(i) #Pull arm i and get reward
        self.estimates[i] += 1. / (self.counts[i]+1) * (reward - self.estimates[i]) #Update estimate for arm i
        return i

# #4. UCB1 solver
class UCB1(Solver): #Upper-confidence bound

    def __init__(self, bandit, init_theta=1.0):
        """
        init_theta (float): We optimistically set initial theta to be 1.0
        """
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_theta]*self.bandit.n #Optimisistic initialisation
        self.name = "UCB1"

    @property
    def estimated_thetas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick best one with consideration of upper confidence bounds
        i = max(range(self.bandit.n), key=lambda x:self.estimates[x]
                + np.sqrt(2*np.log(self.t)/(1+self.counts[x])))
        reward = self.bandit.pull_arm(i)

        self.estimates[i] += 1. / (self.counts[i]+1) * (reward - self.estimates[i]) #Update estimate for arm i

        return i

#5. Thomson solving

class ThompsonSampling(Solver):
    def __init__(self, bandit, init_alpha=1, init_beta=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit)

        self._alphas = [init_alpha] * self.bandit.n
        self._betas = [init_beta] * self.bandit.n
        self.name = "Thompson Sampling"

    @property
    def estimated_probas(self):
        return [self._alphas[i] / (self._alphas[i] + self._betas[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._alphas[x], self._betas[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        reward = self.bandit.pull_arm(i)

        self._alphas[i] += reward
        self._betas[i] += (1 - reward)

        return i

# Let us now test the solvers over 20 trials, each with 1000 steps

def run_solver (solver, trials = 20, iterations = 1000):
    regret_history = []
    for trial in range(trials):
        solver.regret = 0
        solver.regrets = []
        solver.run(iterations)
        regret_history.append(solver.regret)
    print(solver.name, np.mean(regret_history), "+/-",np.std(regret_history))

random = Random(bandit)
greedy = Greedy(bandit,1)
e_greedy = EpsilonGreedy(bandit,0.1) #Set epsilon = 0.1
ucb1 = UCB1(bandit)
thompson = ThompsonSampling(bandit)
for solver in [random,greedy,e_greedy,ucb1,thompson]:
    run_solver(solver)
