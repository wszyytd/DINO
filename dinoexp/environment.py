import math


def select_nearby_objects(objects, center, radius):
    if not math.isfinite(radius) or not 0 < radius <= 100:
        raise ValueError("environment-object radius must be in (0,100] m")
    result = []
    for obj in objects:
        point = obj.transform.location
        coordinates = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(x) for x in coordinates):
            raise ValueError("invalid environment-object location")
        distance = math.dist(coordinates[:2], center[:2])
        if distance <= radius:
            result.append(
                dict(
                    id=int(obj.id),
                    name=str(obj.name),
                    position=list(coordinates),
                    distance_xy_m=distance,
                )
            )
    return result
