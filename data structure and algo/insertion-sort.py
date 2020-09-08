A = [5, 2, 4, 6, 1, 3]

for j in range(1, len(A)):
    key = A[j] 
    i = j - 1
    print(i)
    print(j)
    while i >= 0:
        if key > A[i]:
            A[i + 1] = A[i]
            A[i] = key
            i = i -1
        else:
            break
print(A)
