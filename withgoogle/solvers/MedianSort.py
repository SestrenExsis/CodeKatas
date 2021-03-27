'''
python InteractiveRunner.py python judges/MedianSort.py 0 -- python solvers/MedianSort.py
python InteractiveRunner.py python judges/MedianSort.py 1 -- python MedianSort.py
'''

'''
Test Set 1: N = 10, Q = 300 * T
Test Set 2: N = 50, Q = 300 * T
Test Set 3: N = 50, Q = 170 * T

With a request of A B C:
    if the response is A, then B > A > C or C > A > B
    if the response is B, then A > B > C or C > B > A
    if the response is C, then A > C > B or B > C > A
'''

import collections
import functools
import logging
import sys

logging.basicConfig(
    filename = 'MedianSort.out',
    filemode ='w',
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    )

def query(queries: dict, x: int, y: int, z: int):
    request = ' '.join(sorted([str(x), str(y), str(z)]))
    logging.debug(request)
    if request not in queries:
        print(request)
        sys.stdout.flush()
        response = int(input())
        queries[request] = response
    result = queries[request]
    if result not in (x, y, z):
        logging.debug(result, x, y, z)
        exit()
    return result

def guess(nums: list):
    request = ' '.join(list(map(str, nums)))
    print(request)
    sys.stdout.flush()
    response = int(input())
    result = response
    return result

def solve(element_count: int, query_limit: int):
    queries = {}
    nums = set(range(1, element_count + 1))
    '''
    Find our first pair of adjacent elements (A, B)
    Query cost (worst case) = (N - 1) * (N - 2) / 2
    '''
    logging.debug('Find our first pair of adjacent elements')
    x = nums.pop()
    y = nums.pop()
    while len(nums) > 0:
        z = nums.pop()
        response = query(queries, x, y, z)
        if response == z:
            y = z
    A = x
    B = y
    '''
    Partition all other elements as being on the A side or the B side
    Query cost = N - 2
    '''
    logging.debug('Partition the elements')
    sideA = []
    sideB = []
    nums = set(range(1, element_count + 1))
    nums.remove(A)
    nums.remove(B)
    logging.debug(nums)
    for num in nums:
        response = query(queries, A, B, num)
        # Assume that A and B are truly adjacent to one another
        try:
            assert response != num
        except AssertionError:
            exit()
        if response == A:
            sideA.append(num)
        elif response == B:
            sideB.append(num)
    logging.debug(sideA)
    logging.debug(sideB)
    '''
    Use (A, x, y) queries to sort the A side relative to A
    Query cost (worst case) = N log N, maybe?
    '''
    logging.debug('Sort A side')
    def compareA(x, y):
        response = query(queries, A, x, y)
        comparison = 0
        if response == x:
            comparison = -1
        elif response == y:
            comparison = 1
        return comparison
    sideA = sorted(sideA, key=functools.cmp_to_key(compareA))
    '''
    Use (B, x, y) queries to sort the B side relative to B
    Query cost (worst case) = N log N, maybe?
    '''
    logging.debug('Sort B side')
    def compareB(x, y):
        response = query(queries, B, x, y)
        comparison = 0
        if response == x:
            comparison = -1
        elif response == y:
            comparison = 1
        return comparison
    sideB = sorted(sideB, key=functools.cmp_to_key(compareB))
    '''
    Give our answer
    We have to reverse the direction of one of the sides
    since we sorted each side in opposite directions from one another
    '''
    sorted_nums = list(reversed(sideA)) + [A, B] + sideB
    verdict = guess(sorted_nums)
    if verdict != 1:
        exit()

test_case_count, element_count, query_limit = map(int, input().split())
for i in range(0, test_case_count):
    solve(element_count, query_limit)
