def knapsack_pr(weights, profits, capacity):
    """You are a delivery driver with a limited capacity truck. 
    Your goal is to maximize your earnings by delivering packages to various locations. 
    Each package has a weight and a profit associated with it. 
    However, you can only deliver packages whose total weight does not exceed your truck's capacity."""
    
    n = len(profits)
    
    
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
  
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + profits[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]


weights = eval(input("Enter the weights: "))
profits = eval(input("Enter the profits: "))
capacity = int(input("Enter the capacity: "))

max_profit = knapsack_pr(weights, profits, capacity)
print(f"The maximum profit is: {max_profit}")
