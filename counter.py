from pdb import main
import time
import msvcrt

total = 0
typed = ""
running = True

print("Counting forever... and there's no way to stop it! (or is there?)")

while running:
    total = total + 1
    ticks = 0
    while ticks < 20 and running:
        ticks = ticks + 1
        line = f"\rcount {total} | {typed}"
        print(line, end="", flush=True)
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.isprintable():
                 running = False
        time.sleep(0.05)

print()
print("... How'd you do that?")

echo "# Infinicounter" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/magmaman4/Infinicounter.git
git push -u origin main