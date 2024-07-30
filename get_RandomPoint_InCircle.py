import random
import matplotlib.pyplot as plt

# centerx,centery为圆心坐标
def getRandomPointInCircle(num, radius, centerx, centery):
    samplePoint = []
    for i in range(num):
        while True:
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            if (x ** 2) + (y ** 2) <= (radius ** 2) and (x ** 2) + (y ** 2) != 0:
                samplePoint.append((int(x) + centerx, int(y) + centery))
                break

        # plt.plot(x + centerx, y + centery, '*', color="blue")

    return samplePoint

