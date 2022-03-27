import math
import operator
import sys

test_case_count = int(input())

for i in range(0, test_case_count):
    min_plot_count = int(input())
    prepared_plots = dict()
    expected_values = dict()
    plot_size = int(math.sqrt(min_plot_count)) + 1
    for y in range(2, 2 + plot_size - 1):
        for x in range(2, 2 + plot_size - 1):
            expected_values[(x, y)] = 9
    while True:
        ideal_plot = max(expected_values.items(), key = operator.itemgetter(1))[0]
        plot_to_prepare = list(ideal_plot)
        print(' '.join(str(x) for x in plot_to_prepare))
        sys.stdout.flush()
        x, y = map(int, input().split(' '))
        if (x, y) == (0, 0):
            break
        if (x, y) not in prepared_plots:
            prepared_plots[(x, y)] = 1
            for offset_x in range(-1, 1 + 1):
                for offset_y in range(-1, 1 + 1):
                    offset = (x + offset_x, y + offset_y)
                    if offset in expected_values:
                        expected_values[offset] -= 1
