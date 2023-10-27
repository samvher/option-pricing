import math
import scipy

def bsm_call(k, s, r, sigma, t, div):
    """Price of a call option using the Black-Scholes-Merton model.

    Args:
        k: strike price
        s: current stock price
        r: risk-free rate (as a fraction, e.g. 0.02 for 2%)
        sigma: volatility
        t: time to maturity
        div: dividend yield (as a fraction)

    Returns:
        float: price of the option
    """
    d1 = (math.log(s/k) + (r - div + sigma**2/2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return (math.exp(-div * t) * s * scipy.stats.norm.cdf(d1) -
            math.exp(-r * t) * k * scipy.stats.norm.cdf(d2))

def bsm_call_delta(k, s, r, sigma, t, div):
    """Delta of a call option using the Black-Scholes-Merton model.

    Args:
        k: strike price
        s: current stock price
        r: risk-free rate (as a fraction, e.g. 0.02 for 2%)
        sigma: volatility
        t: time to maturity
        div: dividend yield (as a fraction)

    Returns:
        float: delta of the option
    """
    d1 = (math.log(s/k) + (r - div + sigma**2/2) * t) / (sigma * math.sqrt(t))
    return scipy.stats.norm.cdf(d1)

def bsm_put(k, s, r, sigma, t, div):
    """Price of a put option using the Black-Scholes-Merton model.

    Args:
        k: strike price
        s: current stock price
        r: risk-free rate (as a fraction, e.g. 0.02 for 2%)
        sigma: volatility
        t: time to maturity
        div: dividend yield (as a fraction)

    Returns:
        float: price of the option
    """
    d1 = (math.log(s/k) + (r - div + sigma**2/2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return (math.exp(-r * t) * k * scipy.stats.norm.cdf(-d2) -
            math.exp(-div * t) * s * scipy.stats.norm.cdf(-d1))

def bsm_put_delta(k, s, r, sigma, t, div):
    """Delta of a put option using the Black-Scholes-Merton model.

    Args:
        k: strike price
        s: current stock price
        r: risk-free rate (as a fraction, e.g. 0.02 for 2%)
        sigma: volatility
        t: time to maturity
        div: dividend yield (as a fraction)

    Returns:
        float: delta of the option
    """
    d1 = (math.log(s/k) + (r - div + sigma**2/2) * t) / (sigma * math.sqrt(t))
    return -scipy.stats.norm.cdf(-d1)

def black_call(x, f, r, sigma, t):
    """Price of a call option using Black's model.

    Args:
        x: strike price
        f: current forward price of the bond
        r: spot yield for maturity t
        sigma: volatility of forward bond price
        t: time to option expiration

    Returns:
        float: price of the option
    """
    d1 = (math.log(f/x) + sigma**2/2 * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return math.exp(-r * t) * (f * scipy.stats.norm.cdf(d1) - x * scipy.stats.norm.cdf(d2))

def black_put(x, f, r, sigma, t):
    """Price of a put option using Black's model.

    Args:
        x: strike price
        f: current forward price of the bond
        r: spot yield for maturity t
        sigma: volatility of forward bond price
        t: time to option expiration

    Returns:
        float: price of the option
    """
    d1 = (math.log(f/x) + sigma**2/2 * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return math.exp(-r * t) * (x * scipy.stats.norm.cdf(-d2) - f * scipy.stats.norm.cdf(-d1))

def binom_tree(mu, sigma, r, div, s0, t, k, option_type, n):
    """Binomial tree for a stock and a European call/put option on it.
    
    Args:
        mu: growth rate of the stock
        sigma: volatility of the stock
        r: risk-free rate of the market
        div: dividend yield of the stock
        s0: current price of the stock
        t: time to option expiration
        k: option strike price
        option_type: "call" or "put"
        n: number of steps in the binomial tree
    
    Returns:
        float, float: tree describing stock price evolution, tree describing option price evolution
    """
    assert option_type in ["call", "put"]
    
    h = t / n
    u = math.exp(sigma * math.sqrt(h))
    d = 1 / u
    
    # p = (math.exp((mu - div) * h) - d) / (u - d)
    q_star = (math.exp((r - div) * h) - d) / (u - d)
    
    stock_tree = [[s0]]
    for i in range(1, n+1):
        stock_tree.append([stock_tree[-1][0] * u] + [t * d for t in stock_tree[-1]])
    
    if option_type == "call":
        option_tree = [[max(s - k, 0) for s in stock_tree[-1]]]
    elif option_type == "put":
        option_tree = [[max(k - s, 0) for s in stock_tree[-1]]]
    
    while len(option_tree[0]) >= 2:
        option_tree = [[math.exp(-r * h) * (q_star * option_tree[0][i] +
                                            (1 - q_star) * option_tree[0][i+1])
                        for i in range(0, len(option_tree[0]) - 1)]] + option_tree
    
    return stock_tree, option_tree

def price_binomial(mu, sigma, r, div, s0, t, k, option_type, n):
    """Price of a European option using the binomial tree approximation.

    Args:
        mu: growth rate of the stock
        sigma: volatility of the stock
        r: risk-free rate of the market
        div: dividend yield of the stock
        s0: current price of the stock
        t: time to option expiration
        k: option strike price
        option_type: "call" or "put"
        n: number of steps in the binomial tree

    Returns:
        float: price of the option
    """
    return binom_tree(mu, sigma, r, div, s0, t, k, option_type, n)[1][0][0]

def delta_binomial(mu, sigma, r, div, s0, t, k, option_type, n):
    """Delta of a European option using the binomial tree approximation.

    Args:
        mu: growth rate of the stock
        sigma: volatility of the stock
        r: risk-free rate of the market
        div: dividend yield of the stock
        s0: current price of the stock
        t: time to option expiration
        k: option strike price
        option_type: "call" or "put"
        n: number of steps in the binomial tree

    Returns:
        float: delta of the option
    """
    stock_tree, option_tree = binom_tree(mu, sigma, r, div, s0, t, k, option_type, n)
    return (option_tree[1][0] - option_tree[1][1]) / (stock_tree[1][0] - stock_tree[1][1])