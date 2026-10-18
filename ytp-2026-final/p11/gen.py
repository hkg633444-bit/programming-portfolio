import random,sys
seed=int(sys.argv[1])
random.seed(seed)
n=random.choice([random.randint(1,40),random.randint(60,70),random.randint(120,300)])
mode=random.randint(0,2)
if mode==0:
    R=3;k=random.randint(0,3)
elif mode==1:
    R=10;k=random.randint(0,12)
else:
    R=10**9;k=random.choice([0,random.randint(0,2*10**9),2*10**9,10**9])
print(n,k)
for _ in range(n):
    print(*[random.randint(-R,R) for _ in range(5)])
