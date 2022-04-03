import collections
import math
import sys

'''
Limits
Time limit: 120 seconds
Memory limit: 1 GB

Test Set 1 (Visible Verdict)
1 ≤ T ≤ 100
2 ≤ N ≤ 100_000
K = 8_000
Each room has at least one passage connected to it

Operations
'W'   --> Walk through a random passage in the current room
'T s' --> Teleport to room s
'E e' --> Estimate the cave contains e passages
'''

T = int(input())
for i in range(0, T):
    '''
    N = number of rooms in the cave
    K = maximum number of operations allowed
    R = room number
    P = number of passages connecting to current room
    '''
    N, K = tuple(map(int, input().split(' ')))
    R, P = tuple(map(int, input().split(' ')))
    caves = {}
    caves[R] = P
    passage_counts = collections.defaultdict(int)
    passage_counts[P] += 1
    caves_to_visit = set(range(1, N + 1))
    ops_left = K
    while ops_left > 0:
        # Teleport to one of the unvisited caves
        next_cave = caves_to_visit.pop()
        command = f'T {next_cave}'
        print(command)
        sys.stdout.flush()
        ops_left -= 1
        R, P = tuple(map(int, input().split(' ')))
        passage_counts[P] += 1
        caves[R] = P
    population_estimate = 0
    for passages, count in passage_counts.items():
        multiplier = N * (count / len(caves))
        population_estimate += multiplier * passages
    population_estimate /= 2
    lower_bound = math.floor((2 / 3) * population_estimate)
    upper_bound = math.ceil((4 / 3) * population_estimate)
    E = math.ceil((lower_bound + upper_bound) / 2)
    command = f'E {E}'
    print(command)
    sys.stdout.flush()