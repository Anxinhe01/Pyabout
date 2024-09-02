a = [i for i in range(10)]
for i in a:
    if i in (3, 4):
        a.remove(i)
print(a)