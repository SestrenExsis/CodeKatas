import sys

'''
Smallest 100 numbers total to 5,050
Largest 100 numbers total to 100,999,994,950
Largest power of two allowed is 2 ^ 29 or 536,870,912
'''

T = int(input())
for i in range(T):
    _ = int(input())
    N = 100 # Assume N is always 100
    # Generate valid powers of two
    powers = set()
    for exp in range(30):
        power = 2 ** exp
        powers.add(power)
    # Start a pool
    pool = set()
    # Add powers of two to the pool
    for power in powers:
        pool.add(power)
    num = 1
    while len(pool) < N:
        # Fill up the pool
        if num not in pool:
            pool.add(num)
        num += 1
    # send message
    message = ' '.join(map(str, pool))
    print(message)
    sys.stdout.flush()
    # receive response
    response = tuple(map(int, input().split(' ')))
    pool.union(response)
    left_nums = []
    right_nums = []
    while len(pool) > len(powers):
        next_num = max(num for num in pool if num not in powers)
        pool.remove(next_num)
        left_sum = sum(left_nums)
        right_sum = sum(right_nums)
        left_diff = abs(left_sum + next_num - right_sum)
        right_diff = abs(right_sum + next_num - left_sum)
        if left_diff <= right_diff:
            left_nums.append(next_num)
    target = pool // 2
    