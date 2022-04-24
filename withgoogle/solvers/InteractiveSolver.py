import sys

T = int(input())
for i in range(T):
    # send message
    message = 'HELLO!'
    print(message)
    sys.stdout.flush()
    # receive response
    response = input()
    