class Solver:
    def solve(self, costs: dict, mural: str) -> int:
        cost_c = 0
        cost_j = 0
        for i in range(len(mural)):
            next_cost_c = min(cost_c, cost_j + costs['JC'])
            next_cost_j = min(cost_j, cost_c + costs['CJ'])
            char = mural[i]
            if char in ('C', '?'):
                cost_c = next_cost_c
            else:
                cost_c = float('inf')
            if char in ('J', '?'):
                cost_j = next_cost_j
            else:
                cost_j = float('inf')
        result = min(cost_c, cost_j)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            cost1, cost2, mural = input().split(' ')
            costs = {
                'CJ': int(cost1),
                'JC': int(cost2),
            }
            solution = self.solve(costs, mural)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    solver = Solver()
    solver.main()