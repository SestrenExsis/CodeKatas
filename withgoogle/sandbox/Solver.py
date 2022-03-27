
import collections

class Solver:
    '''
    Minimum people to invite to get everyone to clap
    '''
    def is_standing_ovation(self, shyness_levels: tuple) -> bool:
        standing_ovation = False
        applause_count = 0
        for shyness, count in enumerate(shyness_levels):
            if applause_count >= shyness:
                applause_count += count
        if applause_count >= sum(shyness_levels):
            standing_ovation = True
        result = standing_ovation
        return result

    def solve(self, max_shyness: int, shyness_levels: tuple):
        result = max_shyness
        left = 0
        right = max_shyness
        while left < right:
            mid = left + (right - left) // 2
            modified_shyness_levels = list(shyness_levels)
            modified_shyness_levels[0] += mid
            if self.is_standing_ovation(modified_shyness_levels):
                right = mid
            else:
                left = mid + 1
        result = left
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            parts = input().split(' ')
            max_shyness = int(parts[0])
            shyness_levels = list(map(int, list(parts[1])))
            solution = self.solve(max_shyness, shyness_levels)
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