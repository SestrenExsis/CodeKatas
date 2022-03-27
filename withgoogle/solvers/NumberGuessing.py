
import collections
import functools
import logging
import sys

logging.basicConfig(
    filename = 'NumberGuessingLog.out',
    filemode ='w',
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %I:%M:%S',
    )

class NumberGuessing: # 2018.P.1
    def __init__(self):
        logging.info('Number Guessing')
        
    class Solver:
        def __init__(self, lower_bound, upper_bound, max_tries):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.max_tries = max_tries
            self.tries = 0
            self.queries = {}

        def guess(self, num: int) -> str:
            self.tries += 1
            print(num)
            sys.stdout.flush()
            result = input()
            return result

        def solve(self):
            left = self.lower_bound
            right = self.upper_bound
            while left <= right:
                num = left + (right - left) // 2
                response = self.guess(num)
                if response == 'CORRECT':
                    logging.info(f'The secret number was {num}!')
                    return
                elif response == 'TOO_SMALL':
                    left = num + 1
                elif response == 'TOO_BIG':
                    right = num - 1
                else:
                    logging.error(f'ERROR! Response from judge indicates a problem: {response}')
                    exit()
    
    def main(self):
        T = int(input())
        for _ in range(T):
            lower_bound, upper_bound = tuple(map(int, input().split(' ')))
            max_tries = int(input())
            solver = self.Solver(lower_bound, upper_bound, max_tries)
            solver.solve()

if __name__ == '__main__':
    # python InteractiveRunner.py python NumberGuessingJudge.py 0 -- python NumberGuessingSolver.py
    solver = NumberGuessing()
    solver.main()
