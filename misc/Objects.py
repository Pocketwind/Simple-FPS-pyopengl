import glm

class Objects:
    def __init__(self, parent):
        self.pos=glm.vec3(0,0,0)
        self.velocity=0
        self.angle=0
        self.gravityTimer=0
        self.parent = parent
        self.n=0
        self.children = []
        if parent is not None:
            parent.children.append(self)

        self.GlobalM = glm.mat4() # local to global transformation w.r.t. world frame
        self.LocalM = glm.mat4() # coordinate transformation w.r.t. parent coordinate system
        self.LocalG = glm.mat4() # local shape transformation
        self.color = glm.vec3(1,1,0)

    def SetColor(self,color):
        self.color = color

    def GetColor(self):
        return self.color
    
    def SetLocalM(self, M1):
        self.LocalM = M1

    def GetLocalM(self):
        return self.LocalM

    def SetLocalG(self, G1):
        self.LocalG = G1

    def GetLocalG(self):
        return self.LocalG
    
    def SetGlobalM(self):
        if self.parent is not None:
            self.GlobalM = self.parent.GetGlobalM() * self.LocalM # e.g., M = (M1*M2)*M3
        else:
            self.GlobalM = self.LocalM # e.g., M = M1

        for child in self.children:
            child.SetGlobalM()

    def GetGlobalM(self):
        return self.GlobalM
    
    def GetModelMatrix(self):
        return self.GlobalM * self.LocalG