import time
import msvcrt

total = 0
typed = ""
running = True

poll = 0.05
target = 100
interval = 0.02

next_count = time.time()

print("Don't miss... type any key to stop the counter!")
print("target: 100")
print("speed: every 0.02s")
print("goal: stop within 3 of the target")

while running:
    now = time.time()
    if now >= next_count:
        total = total + 1
        next_count = now + interval
    line = f"\rcount {total} | every {interval:.2f}s    "
    print(line, end="", flush=True)

    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch.isprintable():
            running = False

    time.sleep(poll)

print()
print("Wow, nice! You stopped the counter at", total, "from the target of", target, "only", total - target, "off!")