"""You have an oil tank with a capacity of C litres that 
can be bought and sold by N people. The people 
are standing in a queue are served sequentially in 
the order of array A.
 Some of them want to sell a litre of oil and some of 
them want to buy a litre of oil and A describes this. 
Here, A[i] = 1 denotes that the person wants to sell 
a litre of oil and A[i] = -1 denotes that the person 
wants to buy a litre of oil.
 When a person wants to sell a litre of oil but the 
tank is full, they cannot sell it and become upset. 
Similarly, when a person wants to buy a litre of 
oil but the tank is empty, they cannot buy it and 
become upset. Both these cases cause disturbances.
 You can minimize the disturbance by filling the tank 
initially with a certain X litres of oil.
 Find the minimum initial amount of oil X that results 
in the least number of disturbances.
 Input Format
 The first line contains an integer, N, denoting the 
number of elements in A.
 The next line contains an integer, C, denoting the 
capacity of the tank.
 Each line i of the N subsequent lines (where 0 ≤ i < 
N) contains an integer describing A[i].
 Constraints
 1 <= N <= 10^5
 1 <= C <= 10^5-1 <= A[i] <= 1
 Sample Test Cases
 Case 1
 Input:
 3
 3-1
 1
 1
 External Document © 2025 Infosys Limited 
External Document © 2025 Infosys Limited 
Output:
 1
 Explanation:
 Given N = 3, C = 3, A = [-1, 1, 1].
 To avoid disturbance for Person 1, we need at least 
1 liter in the tank initially.
 After Person 1 buys 1 liter, the tank will be empty.
 Person 2 sells 1 liter, so the tank will have 1 liter.
 Person 3 sells another liter, so the tank will have 2 
liters.
 The minimum initial amount X needed to achieve 
the least number of disturbances is 1 liter.
 Case 2
 Input:
 3
 2-1-1
 1
 Output:
 2
 Explanation:
 Given N = 3, C = 2, A = [-1, -1, 1].
 To ensure that there are no disturbances:
 We need at least 1 liter for Person 1.
 We need an additional 1 liter for Person 2, making 
the total initial amount of oil X = 2.
 Thus, the minimum initial amount of oil X required 
to achieve the least number of disturbances is 2.
 Case 3
 Input:
 4
 3
 1
 1
 1
 1
 1
"""


import sys
import threading

def main():
    import bisect
    input = sys.stdin.readline

    N = int(input())
    C = int(input())
    A = [int(input()) for _ in range(N)]

    # Build prefix sums P; P[0]=0; for i in 1..N, P[i] = sum A[0..i-1]
    P = [0]*(N+1)
    for i in range(1, N+1):
        P[i] = P[i-1] + A[i-1]

    # Collect the prefix‐levels just before each buy or sell
    buys = []   # stores P[i-1] for A[i-1] == -1
    sells = []  # stores P[i-1] for A[i-1] == +1
    for i in range(1, N+1):
        if A[i-1] == -1:
            buys.append(P[i-1])
        else:
            sells.append(P[i-1])

    buys.sort()
    sells.sort()
    total_buys = len(buys)
    total_sells = len(sells)

    # Initial disturbances at X = 0
    #   buy-disturb: whenever Pprefix <= -X = 0  ⇒ buys with Pprefix <= 0
    buy_dist = bisect.bisect_right(buys, 0)
    #   sell-disturb: whenever Pprefix >= C - X = C  ⇒ sells with Pprefix >= C
    sell_dist = total_sells - bisect.bisect_left(sells, C)

    curr = buy_dist + sell_dist
    best = curr
    bestX = 0

    # Build events: at certain X the count changes
    #   For a buy at prefix p, disturbance if X <= -p.
    #   Removing that disturbance happens when X crosses > -p ⇒ at X = -p+1
    #   => event at x = -p+1: curr -= 1
    #   For a sell at prefix p, disturbance if X >= C - p.
    #   That starts when X crosses ≥ C-p ⇒ at X = C-p
    #   => event at x = C-p: curr += 1
    events = {}
    def add_event(x, delta):
        if 0 <= x <= C:
            events[x] = events.get(x, 0) + delta

    for p in buys:
        x0 = -p + 1
        add_event(x0, -1)

    for p in sells:
        x0 = C - p
        add_event(x0, +1)

    # Sweep X from 1 to C over only event points in sorted order
    for x in sorted(events):
        # Skip X=0 since we've already handled it
        if x == 0:
            curr += events[x]
            # check again at X=0 if needed, but no smaller X than 0
            continue
        # Move to X = x
        curr += events[x]
        # Now curr == f(x)
        if curr < best:
            best = curr
            bestX = x
        # tie‑break to smallest X
        elif curr == best and x < bestX:
            bestX = x

    print(bestX)


if __name__ == "__main__":
    # For faster I/O on large N
    threading.Thread(target=main).start()
