'''
python InteractiveRunner.py python judges/InteractiveSolver.py 0 -- python solvers/InteractiveSolver.py
python InteractiveRunner.py python judges/InteractiveSolver.py 1 -- python solvers/InteractiveSolver.py
'''

import collections
import functools
import logging
import sys

logging.basicConfig(
    filename = 'InteractiveSolver.out',
    filemode ='w',
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    )

def log(message: str):
    enable_logging = False
    if enable_logging:
        logging.debug(message)

def send(request: str):
    log('send:' + request)
    print(request)
    sys.stdout.flush()

def receive() -> str:
    response = input()
    log('receive:' + response)
    result = response
    return result

def query(request: str) -> str:
    send(request)
    response = receive()
    result = response
    return result

def solve():
    pass

test_case_count = int(input())
for i in range(0, test_case_count):
    solve()
# At the end of all tests, the judge will issue a verdict
verdict = receive()
log('verdict:' + str(verdict))
exit()