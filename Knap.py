"""
def knapsack_pr(votes, price, budget):
    """political votes counting"""
    
    n = len(price)
    
    
    dp = [[0 for _ in range(budget + 1)] for _ in range(n + 1)]
    
  
    for i in range(1, n + 1):
        for w in range(1, budget + 1):
            if votes[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w -votes[i-1]] + price[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][budget]

def main():
    votes = eval(input("Enter the votes: "))
    price=eval(input("Enter the rate: "))
    budget=int(input("Enter total budget: "))
    print(f"Best choice: {knapsack_pr(votes,price,budget)}")

if __name__=="__main__":
    main()
"""

def delivery(weights, profits, truck_capacity):
    n = len(weights)
    max_profit = 0
    max_combination = []

    for i in range(2**n):
        combination = []
        total_weight = 0
        total_profit = 0

        for j in range(n):
            if (i >> j) & 1:
                combination.append(j + 1)
                total_weight += weights[j]
                total_profit += profits[j]

        if total_weight <= truck_capacity and total_profit > max_profit:
            max_profit = total_profit
            max_combination = combination

    return max_profit, max_combination

weights = []
profits = []
n = int(input("Enter total number of packages : "))
print()
for i in range(n):
    weights.append(int(input(f"Enter weight of package {i+1} : ")))
    profits.append(int(input(f"Enter profit of package {i+1} : ")))
    print()
truck_capacity = 50
max_profit, max_combination = delivery(weights, profits, truck_capacity)
print("Maximum profit:", max_profit)
print("Selected packages:", max_combination)
