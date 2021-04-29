'''
python InteractiveRunner.py python judges/DigitBlocks.py 0 -- python solvers/DigitBlocks.py
python InteractiveRunner.py python judges/DigitBlocks.py 1 -- python DigitBlocks.py
'''

'''
tower_count = 20
blocks_per_tower = 15
Test Set 1: P = 860939810732536850 (approx. 8.6 * 10 ** 17)
Test Set 2: P = 937467793908762347 (approx. 9.37 * 10 ** 17)

300 blocks means, on average
30 0s
30 1s
30 2s
...
30 9s
An ideal pair of towers would look like:
    998877665543210
    987654433221100
    Build a tower of non-9s to height of 14
    First nine you find goes on that tower
    Build a tower of non-9s
The average tower must end in 94 or greater to get both Test Sets
'''

import collections
import functools
import logging
import sys

logging.basicConfig(
    filename = 'DigitBlocks.out',
    filemode ='w',
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    )

def log(message: str):
    enable_logging = False
    if enable_logging:
        logging.debug(message)

def receive() -> int:
    response = int(input())
    log('receive:' + str(response))
    result = response
    return result

def send(tower_id: int):
    request = str(tower_id)
    log('send:' + request)
    print(request)
    sys.stdout.flush()

def solve(tower_count: int, blocks_per_tower: int, target_score: int):
    assert tower_count == 20
    assert blocks_per_tower == 15
    log('tower_count:' + str(tower_count))
    log('blocks_per_tower:' + str(blocks_per_tower))
    log('target_score:' + str(target_score))
    # You can assume that the digit for each block is generated
    # uniformly at random, and independently for each digit
    # There are T * B exchanges
    # Each exchange is:
    #   one receive ('0' to '9'), followed by
    #   one send ('1' to 'T')
    towers = [0] + [blocks_per_tower] * (tower_count)
    block_count = 0
    while block_count < tower_count * blocks_per_tower:
        response = receive()
        log('response:' + str(response))
        if response == -1:
            exit()
        # Figure out the ideal tower to put the block onto
        request = -1
        if response == 9:
            # Put it on the first tower with 1 or more blocks remaining
            for tower_id in range(1, tower_count + 1):
                if towers[tower_id] >= 1:
                    request = tower_id
                    break
        elif response >= 8:
            # Put it on the first tower with 2 or more blocks remaining
            for tower_id in range(1, tower_count + 1):
                if towers[tower_id] >= 2:
                    request = tower_id
                    break
        else:
            # Put it on the first tower with 3 or more blocks remaining
            for tower_id in range(1, tower_count + 1):
                if towers[tower_id] >= 3:
                    request = tower_id
                    break
        if request == -1:
            # Put it on the first tower with 1 or more blocks remaining
            for tower_id in range(1, tower_count + 1):
                if towers[tower_id] >= 1:
                    request = tower_id
                    break
        towers[request] -= 1
        send(request)
        log('request:' + str(request))
        block_count += 1

test_case_count, tower_count, blocks_per_tower, target_score = map(int, input().split())
for i in range(0, test_case_count):
    solve(tower_count, blocks_per_tower, target_score)
# At the end of all tests, the judge will issue a verdict
verdict = receive()
log('verdict:' + str(verdict))
exit()