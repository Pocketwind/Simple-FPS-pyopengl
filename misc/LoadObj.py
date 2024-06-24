import numpy as np

class OBJ:
    def __init__(self, filename, isTexture = False):
        self.ambient = 0.1
        self.diffuse = 0.7
        self.specular = 0.6
        self.shininess = 100
        self.vertices, self.triconnect,\
        self.texture, self.tritexture,\
        self.normals, self.trinormal = self.LoadObjfile(filename, isTexture)

        self.trivertex = self.vtoev(self.vertices, self.triconnect)
        if isTexture:
            self.tritexture = self.vtoev(self.texture, self.tritexture)
        self.trinormal = self.vtoev(self.normals, self.trinormal)
        self.polygons = len(self.triconnect)


    def LoadObjfile(self, ifile, isTexture):
        vertices = []
        texture = []
        normals = []
        triconnect = []
        trinormal = []
        tritexture = []
        with open(ifile,encoding="utf-8") as fp:
            line = fp.readline()
            while line:
                if line[:2] == "v ":
                    vx, vy, vz = [float(value) for value in line[2:].split()]
                    vertices.append((vx, vy, vz))
                if line[:2] == "vt" and isTexture:
                    tx, ty, tz = [float(value) for value in line[2:].split()]
                    texture.append((tx, ty, tz))
                if line[:2] == "vn":
                    nx, ny, nz = [float(value) for value in line[2:].split()]
                    normals.append((nx, ny, nz))        
                if line[:2] == "f ":
                    t1, t2, t3 = [value for value in line[2:].split()]
                    triconnect.append([int(value) for value in t1.split('/')][0] - 1) 
                    triconnect.append([int(value) for value in t2.split('/')][0] - 1)
                    triconnect.append([int(value) for value in t3.split('/')][0] - 1)

                    tritexture.append([int(value) for value in t1.split('/')][1] - 1) 
                    tritexture.append([int(value) for value in t2.split('/')][1] - 1)
                    tritexture.append([int(value) for value in t3.split('/')][1] - 1)

                    trinormal.append([int(value) for value in t1.split('/')][2] - 1) 
                    trinormal.append([int(value) for value in t2.split('/')][2] - 1)
                    trinormal.append([int(value) for value in t3.split('/')][2] - 1)
                line = fp.readline()

        vertices = np.array(vertices, dtype=np.float32) # 4bytes* nvert
        triconnect = np.array(triconnect, dtype=np.uint32) # 4bytes* ntriconnect
        texture = np.array(texture, dtype=np.float32) # 4bytes* nvert
        tritexture = np.array(tritexture, dtype=np.uint32) # 4bytes* ntriconnect
        normals = np.array(normals, dtype=np.float32) # 4bytes* nvert
        trinormal = np.array(trinormal, dtype=np.uint32) # 4bytes* ntriconnect
        
        return vertices, triconnect, texture, tritexture, normals, trinormal

    def vtoev(self, vert, conn):
        tri = []
        for t in range(0, len(conn),3):
            tri.append(vert[conn[t]])
            tri.append(vert[conn[t+1]])
            tri.append(vert[conn[t+2]])
        return np.array(tri,np.float32)