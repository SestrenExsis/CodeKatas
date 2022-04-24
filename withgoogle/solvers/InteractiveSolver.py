import sys

T = int(input())
for i in range(T):
    attempts_left = 300
    message = '11111111'
    print(message)
    sys.stdout.flush()
    attempts_left -= 1
    response = int(input())
    while response > 0 and attempts_left > 0:
        if response >= 5:
            # guarantee a total inversion
            message = '11111111'
            print(message)
            sys.stdout.flush()
            attempts_left -= 1
            # receive response
            response = int(input())
        else:
            # guess
            message = '1' * response + '0' * (8 - response)
            print(message)
            sys.stdout.flush()
            attempts_left -= 1
            # receive response
            response = int(input())
    