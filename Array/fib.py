class Fibonacci:
    def __init__(self):
        self.memo = {}

    def fib(self, n):
        if n in self.memo:
            return self.memo[n]
        if n <= 2:
            return 1
        self.memo[n] = self.fib(n - 1) + self.fib(n - 2)
        return self.memo[n]
    
    def test(self):
        assert self.fib(1) == 1, "Test case 1 failed"
        assert self.fib(2) == 1, "Test case 2 failed"
        assert self.fib(3) == 2, "Test case 3 failed"
        assert self.fib(4) == 3, "Test case 4 failed"
        assert self.fib(5) == 5, "Test case 5 failed"
        assert self.fib(6) == 8, "Test case 6 failed"
        assert self.fib(7) == 13, "Test case 7 failed"
        print("All test cases passed!")
    
def main():
    fib = Fibonacci()
    fib.test()
    print("Fibonacci:", fib.fib(n= int(input("Enter a positive integer: "))))

if __name__ == "__main__":
    main()


