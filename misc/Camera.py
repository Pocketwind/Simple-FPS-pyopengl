from dataclasses import dataclass
import glm

@dataclass
class Camera:
    shoot:bool = False
    grab:bool = False
    grabbedObject:int = 0
    clip:bool = True
    cam:int = 0
    distance:float = 0
    Up:glm.vec3 = glm.vec3(0,1,0)
    Pos:glm.vec3 = glm.vec3(0,0,0)
    Target:glm.vec3 = glm.vec3(0,0,0)
    FOV:float = 45
    Height:int = 0
    Width:int = 0
    aspectRatio:float = 0
    near:float = 0
    far:float = 0
    angle1:float = 0
    angle2:float = 0
    viewVector:glm.vec3 = glm.vec3(0,0,0)
    armPos:glm.vec3 = glm.vec3(0,0,0)
    gravityTimer:int = 0
    tall:int = 3