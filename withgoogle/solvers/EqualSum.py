import logging
import sys

logging.basicConfig(
    filename = 'InteractiveProblem.out',
    filemode ='w',
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    )

'''
Smallest 100 numbers total to 5,050
Largest 100 numbers total to 100,999,994,950
Largest power of two allowed is 2 ^ 29 or 536,870,912
'''

logging.debug('EqualSum')

T = int(input())
for i in range(T):
    logging.debug(f'Test #{i + 1}')
    N = int(input())
    assert N == 100
    logging.debug(f'  N={N}')
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
    logging.debug(f'  message={message}')
    # receive response
    response = input()
    logging.debug(f'  response={response}')
    judge_pool = tuple(map(int, response.split(' ')))
    pool = pool.union(judge_pool)
    logging.debug(f'  total_sum={sum(pool)}')
    left_nums = []
    right_nums = []
    while len(pool) > len(powers):
        next_num = max(num for num in pool if num not in powers)
        pool.remove(next_num)
        left_sum = sum(left_nums)
        right_sum = sum(right_nums)
        if left_sum < right_sum:
            left_nums.append(next_num)
        else:
            right_nums.append(next_num)
    while len(pool) > 0:
        next_num = max(num for num in pool)
        pool.remove(next_num)
        left_sum = sum(left_nums)
        right_sum = sum(right_nums)
        if left_sum < right_sum:
            left_nums.append(next_num)
        else:
            right_nums.append(next_num)
    left_sum = sum(left_nums)
    right_sum = sum(right_nums)
    logging.debug(f'  left_sum={left_sum}')
    logging.debug(f'  right_sum={right_sum}')
    # assert sum(left_nums) == sum(right_nums)
    message = ' '.join(map(str, left_nums))
    print(message)
    sys.stdout.flush()
    