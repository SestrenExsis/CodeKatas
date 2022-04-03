
N=input
for t in range(int(N())):
    N();s=1
    for n in sorted(map(int,N().split())):
        if n>=s:s+=1
    print(f"Case #{t+1}: {s-1}")
