class WarmupCounter:
    def __init__(self, warmup_rounds: int):
        self.warmup_rounds = warmup_rounds
        self.counter = 0

    def update(self):
        if self.counter < self.warmup_rounds:
            self.counter += 1

    def reset(self):
        self.counter = 0

    def is_warmed_up(self):
        return self.counter == self.warmup_rounds