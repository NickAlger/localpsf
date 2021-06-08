import fenics
import mshr

def circle_mesh(center, radius, resolution):
    outer_circle = mshr.Circle(fenics.Point(center[0], center[1]), radius)
    mesh = mshr.generate_mesh(outer_circle, resolution)
    return mesh