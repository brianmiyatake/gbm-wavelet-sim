# include necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import statistics

# fixed seed for reproducibility
rng = np.random.default_rng(0)

''' 
Following the wavelet construction of Brownian motion from Michael Steele's Stochastic Calculus
and Financial Applications
'''

# Define \Delta(x) as the integral of the mother wavelet, \int H(t)\mathrm{d}t
def MotherDelta(x):
    if x < 0 or x > 1:
        return 0
    elif 0 <= x < 1/2:
        return 2 * x
    else:
        return 2 * (1 - x)

# Extend this to the integrals of the entire family of wavelets, defined as $\Delta_n(t)=\Delta(2^jt - k)$,
# where $n = 2^j + k$ with $0\leq k < 2^j$. Further, we take $\Delta_0(t)=t$. 
def Delta(n, x):
    if n == 0:
        return x
    j2 = 1 << math.floor(math.log(n, 2))
    k = n - j2
    return MotherDelta(j2 * x - k)

# We now need the coefficients in the series expansion, which are defined as $\Lambda_0=1$ and
# $\Lambda_n=\frac{1}{2} 2^{-j/2}$, where $n = 2^j + k$ with $0\leq k < 2^j$.
def Lambda(n):
    if n == 0:
        return 1
    j = math.floor(math.log(n, 2))
    return (1/2) * math.pow(2, -j / 2)

'''
We now construct the series expansion of standard Brownian motion, truncated to N summands.
'''

# If $\{Z_n\}_{n=0}^\infty$ is a sequence of independent Gaussian RVs with mean 0 and variance 1, then
# X_t=\sum_{n=0}^\infty \lambda_nZ_n\Delta_n(t)
# converges uniformly on [0,1] a.e. and limit is standard Brownian motion for 0 <= t <= 1

# Constructs standard Brownian motion on the interval [0, T] (T = xpoints[-1]) using the infinite series expansion
# by evaluating the series truncated to N summands at each time in xpoints. We use the theorem above
# and appeal to the fact that the scaled process $\sqrt{T} Y_{t/T}$ is standard Brownian motion on [0, 1].

# Returns an array ypoints that corresponds to B_t for each t in xpoints.
def BrownianMotion(xpoints, numSummands):
    ypoints = []
    Z = rng.normal(0, 1, numSummands)
    for x in xpoints:
        scaledX = x / xpoints[-1]
        tempFactor = math.sqrt(xpoints[-1])
        temp = 0
        for n in range(numSummands):
            temp += tempFactor * Lambda(n) * Z[n] * Delta(n, scaledX)
        ypoints.append(temp)
    return ypoints

# test BrownianMotion function
T = 10
numPoints = 128
numSummands = 128
xpoints = np.linspace(0, T, numPoints)
ypoints = BrownianMotion(xpoints, numPoints)
plt.plot(xpoints, ypoints)
plt.title("One path of standard Brownian motion")
plt.xlabel("Time 0 <= t <= T")
plt.ylabel("Value")
plt.show()

'''
Using the construction of Brownian motion above, we now construct geometric Brownian motion.
'''

# Geometric Brownian motion is defined by the SDE
# $dX_t=\mu X_t dt+\sigma X_t\mathrm{d}B_t$ with $X_0=x_0>0$
# An application of Ito's formula with $X_t=f(t, B_t)$ and matching coefficients with the above SDE
# allow us to solve to obtain $X_t=x_0\exp((\mu-\frac{1}{2}\sigma^2)t+\sigma B_t)$
# Uniqueness of this solution is guaranteed by the space-variable Lipschitz and spatial growth condition, 
# alongside the fact that the above is bounded in L^2(dP). 

# Returns an array of ypoints for each time in xpoints representing one path in geometric Brownian
# motion with X_0 = x0 and parameters mu (drift coefficient) and sigma (volatility coefficient) as above.
def GeometricBrownianMotion(xpoints, numSummands, x0, mu, sigma):
    # temporarily store points of B_t inside ypoints
    ypoints = BrownianMotion(xpoints, numSummands)
    for i in range(len(xpoints)):
        # modify ypoints in-place into GBM
        ypoints[i] = x0 * math.exp((mu - 0.5 * math.pow(sigma, 2)) * xpoints[i] + sigma * ypoints[i])
    return ypoints

# test GeometricBrownianMotion function
T = 10
P = 128
xpoints = np.linspace(0, T, P)
ypoints = GeometricBrownianMotion(xpoints, 256, 10, 0.03, 0.05)
plt.plot(xpoints, ypoints)
plt.title("One path of geometric Brownian motion")
plt.xlabel("Time 0 <= t <= T")
plt.ylabel("Value")
plt.show()

'''
Assuming a stock follows geometric Brownian motion, we can calculate a theoretical price of a
European call option using the Black-Scholes equation.
'''
def BlackScholes(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma):
    d1 = (math.log(initialStockPrice / strikePrice) + (riskFreeRate + (1 / 2) * sigma ** 2) * totalTime) / (sigma * math.sqrt(totalTime))
    d2 = d1 - sigma * math.sqrt(totalTime)
    return initialStockPrice * norm.cdf(d1) - strikePrice * math.exp(-riskFreeRate*totalTime) * norm.cdf(d2)

# test function
# print(BlackScholes(10, 10, 12, 0.03, 0.05))

'''
Create a Monte-Carlo simulation of numerous geometric Brownian motion paths with the same parameters
to simulate the average gain/loss of buying the European call option at some price callPrice
'''

# Returns the payoff of a call option with strike price strikePrice at current price currentPrice.
def payoff(currentPrice, strikePrice):
    return max(0.0, currentPrice - strikePrice)

# returns the simulated average profit for 1 call bought at callPrice.
def MonteCarloCallSim(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma, callPrice, numTrials, numSummands, plot = False):
    # Since our GeometricBrownianMotion function takes in numPoints as a parameter, we can optimize 
    # the speed of calculation for our payoff by not using the function when plotting is False.
    # Instead, we view B_totalTime as N(0.0, totalTime) and plug this into our closed-form for GBM.
    payoffs = []
    if not plot:
        for _ in range(numTrials):
            Z = rng.normal(0.0, math.sqrt(totalTime))
            y = initialStockPrice * math.exp((riskFreeRate - 0.5 * math.pow(sigma, 2)) * totalTime + sigma * Z)
            payoffs.append(payoff(y, strikePrice))
    else:
        numPoints = 50
        xpoints = np.linspace(0, totalTime, numPoints)
        for _ in range(numTrials):
            ypoints = GeometricBrownianMotion(xpoints, numSummands, initialStockPrice, riskFreeRate, sigma)
            # ypoints[-1] is the stock price at maturation time T
            payoffs.append(payoff(ypoints[-1], strikePrice))
            if plot:
                plt.plot(xpoints, ypoints)
        plt.title("Monte-Carlo simulation of GBM with wavelet construction")
        plt.xlabel("Time elapsed (years)")
        plt.ylabel("Value of path in GBM")
        plt.show()
        
    # include the discount factor, because we the payoffs only come after maturation time.
    return payoffs, math.exp(-riskFreeRate * totalTime) * np.mean(payoffs) - callPrice

# test MonteCarloCallSim
MonteCarloCallSim(10, 10, 12, 0.03, 0.05, 1.6, 100, 50, True)

# Notice how if the callPrice is placed below the price outputted by the Black Scholes equation,
# we gain a net profit most of the time because the call is on "discount" relative to its theoretical price.

''' 
What is the expected value of the gain/loss for a specific call price? 
On average, it should be the Black-Scholes theoretical price minus the call price
'''

# We check the above with our Monte-Carlo simulation.
initialStockPrice = 10
totalTime = 10
strikePrice = 12
riskFreeRate = 0.03
sigma = 0.05
numPoints = 50
callPrice = 1.5
numTrials = 1000
numSummands = 50
ScholesMinusCallPrice = BlackScholes(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma) - callPrice
payoffVector, simAvgProfit = MonteCarloCallSim(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma, callPrice, numTrials, numSummands, False)
print(f"The gain/loss from buying at the Black‑Scholes price is {ScholesMinusCallPrice}.")
print(f"The gain/loss from the Monte‑Carlo simulation buying at callPrice is {simAvgProfit}.")
print(f"The error between the Monte-Carlo simulation with the wavelet construction and Black-Scholes is {abs(ScholesMinusCallPrice - simAvgProfit)}.")

# We analyze the expected error of our payoffs to quantify how good the Monte-Carlo simulation is.
samplePayoffVar = statistics.variance(payoffVector) # sample variance
samplePayoffSD = math.sqrt(samplePayoffVar) # standard deviation of payoffs
newSamplePayoffSD = samplePayoffSD / math.sqrt(numTrials) # apply Central Limit Theorem to numTrials
newSamplePayoffSD *= math.exp(-riskFreeRate * totalTime) # apply discount factor, because payoffs are realized in totalTime years
print(f"The error of the Monte-Carlo simulation with the wavelet construction should converge in distribution to N({0.0}, {newSamplePayoffSD ** 2}) by CLT.")
print("\n")

'''
We can also work with the stochastic differential equation dX_t= mu X_t dt + sigma X_t dB_t$ with $X_0=x_0>0$
in the sense that we treat the symbols $dX_t$, $dt$, and $dB_t$ as small but finite changes in $X_t$, $t$, and $B_t$,
respectively. This serves as the motivation for the Euler-Maruyama method for approximating the solutions to stochastic differential equations.
'''

# The true solution X to the above SDE is approximated by the Markov chain {Y}, where we approximate the above SDE with the relation
# $Y_{n+1} - Y_n = \mu Y_n(t_{n + 1} - t_n) + \sigma Y_n(B_{t_{n + 1}} - B_{t_n})$ for a partition $0=t_0 < t_1 < \cdots < t_n = T$ of [0, T]
# into N + 1 equal subintervals.

def GBMEulerMaruyamaApprox(x0, totalTime, riskFreeRate, sigma, numPoints, plot = False):
    Y = [0 for _ in range(numPoints)]
    Y[0] = x0
    dt = totalTime / (numPoints - 1)
    for n in range(numPoints - 1):
        Y[n + 1] = Y[n] + riskFreeRate * Y[n] * dt + sigma * Y[n] * rng.normal(0, math.sqrt(dt))
    if plot:
        plt.plot(np.linspace(0, totalTime, numPoints), Y)
        plt.title("One path of geometric Brownian motion")
        plt.xlabel("Time 0 <= t <= T")
        plt.ylabel("Value")
        plt.show()
    return Y

# test GBMEulerMaruyamaApprox function
GBMEulerMaruyamaApprox(10, 10, 0.03, 0.05, 100, True)

'''
Mirror Monte-Carlo approach with the Euler-Maruyama approach.
'''

def EulerMonteCarloCallSim(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma, callPrice, numTrials, numPoints, plot = False):
    payoffs = []
    for _ in range(numTrials):
        ypoints = GBMEulerMaruyamaApprox(initialStockPrice, totalTime, riskFreeRate, sigma, numPoints)
        if plot:
            plt.plot(np.linspace(0, totalTime, numPoints), ypoints)
        payoffs.append(payoff(ypoints[-1], strikePrice))
    if plot:
        plt.title("Monte-Carlo simulation of GBM with Euler-Maruyama")
        plt.xlabel("Time elapsed (years)")
        plt.ylabel("Value of path in GBM")
        plt.show()
    return payoffs, math.exp(-riskFreeRate * totalTime) * np.mean(payoffs) - callPrice

numPoints = 100
ScholesMinusCallPrice = BlackScholes(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma) - callPrice
payoffVector, simAvgProfit = EulerMonteCarloCallSim(initialStockPrice, totalTime, strikePrice, riskFreeRate, sigma, callPrice, numTrials, numPoints, True)
print(f"The gain/loss from buying at the Black‑Scholes price is {ScholesMinusCallPrice}.")
print(f"The gain/loss from the Monte‑Carlo simulation buying at callPrice with the Euler‑Maruyama approach is {simAvgProfit}")
print(f"The error between the Monte-Carlo simulation with the Euler-Maruyama approach and Black-Scholes is {abs(ScholesMinusCallPrice - simAvgProfit)}.")
# We analyze the error similarly to the other Monte-Carlo approach, using the same parameters.
samplePayoffVar = statistics.variance(payoffVector) # sample variance
samplePayoffSD = math.sqrt(samplePayoffVar) # standard deviation of payoffs
newSamplePayoffSD = samplePayoffSD / math.sqrt(numTrials) # apply Central Limit Theorem to numTrials
newSamplePayoffSD *= math.exp(-riskFreeRate * totalTime) # apply discount factor, because payoffs are realized in totalTime years
print(f"The error of the Monte-Carlo simulation with the Euler-Maruyama approach should converge in distribution to N({0.0}, {newSamplePayoffSD ** 2}) by CLT.")


