import math



def generate_circle(r=1, n=10):
    return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n)]