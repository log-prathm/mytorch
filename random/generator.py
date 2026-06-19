class Generator:
    def __init__(self, seed=46):
        self.state = seed

    def next_uint32(self):
        self.state = (
            self.state * 1664525
            + 1013904223
        ) % (2**32)
        return self.state 

    def rand(self):
        return self.next_uint32() / (2**32)

    
default_generator = Generator()

if __name__ == "__main__":
    for s in [1,2,3, 1000]:
        r = Generator(s)
        print(s, "->", r.rand())