N = 10

m = 31
a = 4
x = 3
for i in range(N):
    x = (a * x) % m
    print(x)