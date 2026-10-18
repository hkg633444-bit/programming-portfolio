import random,sys
seed=int(sys.argv[1])
random.seed(seed)
n=random.randint(2,6)
m=random.randint(1,10)
k=random.randint(0,min(m,4))
mx=random.choice([3,10,10**9])
print(n,m,k)
for _ in range(m):
    u=random.randint(1,n)
    v=random.randint(1,n)
    while v==u:
        v=random.randint(1,n)
    print(u,v,random.randint(1,mx))
