import task_generator
import tsp

circle_coordinates = task_generator.generate_circle(1, 30)
problem = tsp.from_x_y(circle_coordinates)
print(problem)