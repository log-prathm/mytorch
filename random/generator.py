class Generator:
    def __init__(self, seed):
        self.state = seed

    def rand(self):
        self.state = (
            self.state * 1664525
            + 1013904223
        ) % (2**32)
        print(self.state)
        return self.state / (2**32)

if __name__ == "__main__":
    for s in [1,2,3, 1000]:
        r = Generator(s)
        print(s, "->", r.rand())