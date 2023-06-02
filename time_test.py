import time


t0 = time.time()

for i in range(10000):
    time.sleep(0.5)
    print(f'\r{time.time() - t0:.2f} s have passed', end='')