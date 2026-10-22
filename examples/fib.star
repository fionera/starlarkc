def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

values = [fib(i) for i in range(15)]
print(values)

squares = {str(i): i * i for i in range(6)}
print(squares)
