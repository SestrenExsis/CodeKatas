'''
python InteractiveRunner.py python judges/ESAbATAd.py 0 -- python solvers/ESAbATAd.py
python InteractiveRunner.py python judges/ESAbATAd.py 1 -- python ESAbATAd.py
'''

'''
1, 11, 21, 31, ...
25% chance nothing happens
25% chance 0s and 1s are complemented
25% chance array is reversed
25% chance both complemented and reversed

abcdefghij
ABCDEFGHIJ
jihgfedcba
JIHGFEDBCA

first outside pair that matches is your key for determining if bits have been complemented
first outside pair that doesn't match is your key for determining if bits have been reversed

if aj initially == 00
    a == 0 means bits have not been complemented
    a == 1 means bits have been complemented
if aj initially == 11
    a == 0 means bits have been complemented
    a == 1 means bits have not been complemented

00000 ... 00000 --> f[2 ... 10] = f[1]
00000 ... 11111
11111 ... 00000
11111 ... 11111

get f(0), f(1), f(2), f(3), f(4), f(n - 0), f(n - 1), f(n - 2), f(n - 3), f(n - 4)
values = [
    (f(1), 1 - f(1), f(n - 0), 1 - f(n - 0)),
    (f(2), 1 - f(2), f(n - 1), 1 - f(n - 1)),
    (f(3), 1 - f(3), f(n - 2), 1 - f(n - 2)),
    (f(4), 1 - f(4), f(n - 3), 1 - f(n - 3)),
    (f(5), 1 - f(5), f(n - 4), 1 - f(n - 4)),
    ...
    (f(n - 4), 1 - f(n - 4), f(5), 1 - f(5)),
    (f(n - 3), 1 - f(n - 3), f(4), 1 - f(4)),
    (f(n - 2), 1 - f(n - 2), f(3), 1 - f(3)),
    (f(n - 1), 1 - f(n - 1), f(2), 1 - f(2)),
    (f(n - 0), 1 - f(n - 0), f(1), 1 - f(1)),
    ]
    
after every ten requests, it will take 1 or 2 attempts to correctly update the lookups
depending on what has been observed so far
'''

import sys

def query(queries, request):
    if request is None:
        request = 1
    elif type(request) == int:
        request += 1
    print(request)
    sys.stdout.flush()
    response = input()
    result = (request, response)
    queries.append(result)
    return result

def solve(bit_count: int):
    queries = []
    bits = [0] * bit_count
    flip_index = None
    reverse_index = None
    for i in range(bit_count // 2):
        if len(queries) > 0 and len(queries) % 10 == 0:
            flip_check = query(queries, flip_index)[1]
            reverse_check = query(queries, reverse_index)[1]
            if (
                flip_index is not None and
                bits[flip_index] ^ int(flip_check)
                ):
                # Flip each bit from 0 to 1 or from 1 to 0
                for index in range(len(bits)):
                    bits[index] ^= 1
            if (
                reverse_index is not None and
                bits[reverse_index] ^ int(reverse_check)
                ):
                bits.reverse()
        j = bit_count - i - 1
        bits[i] = int(query(queries, i)[1])
        bits[j] = int(query(queries, j)[1])
        if bits[i] == bits[j]:
            flip_index = i
        else:
            reverse_index = i
    guess = ''.join(map(str, bits))
    verdict = query(queries, guess)[1]
    if verdict != 'Y':
        exit()

test_case_count, bit_count = map(int, input().split())
for i in range(0, test_case_count):
    solve(bit_count)
