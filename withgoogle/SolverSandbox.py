import collections

class Solver:
    def get_hits(self, K, P, a, b):
        hits = set()
        for card in range(1, K + 1):
            left = card
            right = card
            while left >= 0 or right <= K:
                if left in P or right in P:
                    break
                if left in (a, b) or right in (a, b):
                    hits.add(card)
                    break
                left -= 1
                right += 1
        result = len(hits)
        return result
    
    def bruteforce(self, K, P):
        assert K <= 30
        max_odds = 0
        for a in range(1, K + 1):
            if a in P:
                continue
            for b in range(1, K + 1):
                if b in P:
                    continue
                hits = self.get_hits(K, P, a, b)
                odds = hits / K
                max_odds = max(max_odds, odds)
        result = max_odds
        return result
    
    def solve(self, K, P):
        return 0
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            N, K = tuple(map(int, input().split(' ')))
            P = set(tuple(map(int, input().split(' '))))
            solution = self.bruteforce(K, P)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    '''
    Usage
    python SolverSandbox.py < inputs/SolverA.in
    '''
    solver = Solver()
    solver.main()