import collections

class Solver:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
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