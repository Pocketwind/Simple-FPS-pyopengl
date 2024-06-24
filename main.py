from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 
import glm, time
from OpenGL.GL.shaders import compileProgram,compileShader
from misc.ArmData import *
from misc.Camera import *
from misc.LoadObj import *
from misc.Objects import *

TITLE = "테스트"
TARGET_FPS = 120
SCREEN_WIDTH,SCREEN_HEIGHT = 1600,1200
INITIAL_X,INITIAL_Y,INITIAL_Z = -5,10,5
PLAYER_SPEED = 0.3
MOUSE_SPEED=0.001
COLLIDE_THRESHOLD=2
ARM_DISTANCE=3
GRAVITY_STEP=0.01*60/TARGET_FPS

spheres=[Objects(None) for _ in range(4)]
character=Objects(None)
objects=spheres+[character]
cam=Camera()

def key_callback(window, key, scancode, action, mods):
    global cam
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    if key==GLFW_KEY_C and action==GLFW_PRESS:
        cam.clip=not cam.clip
    if key==GLFW_KEY_E and action==GLFW_PRESS:
        collision,_=CheckObjectsCollision(objects,cam.armPos)
        if len(collision)!=0:
            cam.grab=not cam.grab
            cam.grabbedObject=collision[0]
        print(collision,cam.grab,cam.grabbedObject)
    if key==GLFW_KEY_LEFT_CONTROL:
        cam.Pos-=glm.vec3(0,1,0)*PLAYER_SPEED
    if key==GLFW_KEY_LEFT_SHIFT:
        cam.Pos+=glm.vec3(0,1,0)*PLAYER_SPEED
    if key==GLFW_KEY_SPACE and action==GLFW_PRESS:
        cam.shoot=True
    #클립
    if cam.clip:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_D:
                cam.Pos+=glm.normalize(glm.cross(cam.viewVector,cam.Up))*PLAYER_SPEED
            if key==GLFW_KEY_A:
                cam.Pos-=glm.normalize(glm.cross(cam.viewVector,cam.Up))*PLAYER_SPEED
            if key==GLFW_KEY_S:
                cam.Pos-=glm.normalize(cam.viewVector*glm.vec3(1,0,1))*PLAYER_SPEED
            if key==GLFW_KEY_W:
                cam.Pos+=glm.normalize(cam.viewVector*glm.vec3(1,0,1))*PLAYER_SPEED
    #노클립
    else:
        cam.gravityTimer=0
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_D:
                cam.Pos+=glm.normalize(glm.cross(cam.viewVector,cam.Up))*PLAYER_SPEED*2
            if key==GLFW_KEY_A:
                cam.Pos-=glm.normalize(glm.cross(cam.viewVector,cam.Up))*PLAYER_SPEED*2
            if key==GLFW_KEY_S:
                cam.Pos-=cam.viewVector*PLAYER_SPEED*2
            if key==GLFW_KEY_W:
                cam.Pos+=cam.viewVector*PLAYER_SPEED*2

def cursor_callback(window, xpos, ypos):
    global cam
    cam.angle1-=MOUSE_SPEED*(SCREEN_WIDTH/2-xpos)
    cam.angle2+=MOUSE_SPEED*(SCREEN_HEIGHT/2-ypos)
    if cam.angle2>np.pi/2:
        cam.angle2=np.pi/2-1e-6
    if cam.angle2<-np.pi/2:
        cam.angle2=-np.pi/2+1e-6

def main():
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
    window = glfwCreateWindow(SCREEN_WIDTH,SCREEN_HEIGHT,TITLE,None,None)
    if not window:
        glfwTerminate()
        return Exception("GLFW 창 생성 실패")
    
    glfwMakeContextCurrent(window)
    glfwSetKeyCallback(window,key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED)

    
    sphereModelPath="models/sphere.obj"
    cubeModelPath="models/cube.obj"
    characterModelPath="models/paimon.obj"
    sphereModel = OBJ(sphereModelPath)
    cubeModel = OBJ(cubeModelPath)
    characterModel = OBJ(characterModelPath)
    sphereModel.ambient = 0.1
    sphereModel.diffuse = 0.9
    sphereModel.specular = 1
    sphereModel.shininess = 120
    cubeModel.ambient = 0.3
    cubeModel.diffuse = 0.5
    cubeModel.specular = 0.1
    cubeModel.shininess = 100
    characterModel.ambient = 0.8
    characterModel.diffuse = 0.4
    characterModel.specular = 0.2
    characterModel.shininess = 100

    vaoSphere = InitializeObject(sphereModel)
    vaoCube = InitializeObject(cubeModel)
    vaoFrame = InitializeWorldFrame()
    vaoCharacter = InitializeObject(characterModel)

    shader = CreateShader(vertexShaderSrc_frame, fragmentShaderSrc_frame)
    AmatLoc = glGetUniformLocation(shader, 'Amat')

    shader_Object = CreateShader(vertexShaderSrc_object,fragmentShaderSrc_object)
    AmatLocObject = glGetUniformLocation(shader_Object, 'Amat')
    MmatLocObject = glGetUniformLocation(shader_Object, 'Mmat')
    camVecLocObject = glGetUniformLocation(shader_Object, 'camVec')
    vertexColorLocObject = glGetUniformLocation(shader_Object, 'vertexColor')
    ambientLocObject = glGetUniformLocation(shader_Object, 'ambientStrength_in')
    specularLocObject = glGetUniformLocation(shader_Object, 'specularStrength_in')
    shininessLocObject = glGetUniformLocation(shader_Object, 'shininess_in')
    diffuseLocObject = glGetUniformLocation(shader_Object, 'diffuseStrength_in')

    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)

    #cam 초기화
    #cam = Camera()
    cam.cam=0
    cam.distance=20
    cam.Up,cam.Target=glm.vec3(0,1,0),glm.vec3(0,0,0)
    cam.Pos=glm.vec3(INITIAL_X,INITIAL_Y,INITIAL_Z)
    cam.FOV=glm.radians(45)
    cam.Height,cam.Width=SCREEN_HEIGHT,SCREEN_WIDTH
    cam.aspectRatio=cam.Width/cam.Height
    cam.near=0.1
    cam.far=100
    cam.angle1,cam.angle2=0,0

    #sphere=Objects(None)
    for i in range(4):
        spheres[i].tall=1
        spheres[i].n=i+1
        spheres[i].SetColor(glm.vec3(1,1,1))
        spheres[i].SetLocalM(glm.translate(glm.vec3(-5+i*4,3,-5+i*4)))
        spheres[i].pos=glm.vec3(-5+i*4,3,-5+i*4)
        spheres[i].SetLocalG(glm.scale(glm.vec3(1,1,1)))
        spheres[i].SetGlobalM()
    

    floor=Objects(None)
    floor.SetLocalM(glm.translate(glm.vec3(0,-0.1,0)))
    floor.SetColor(glm.vec3(189/255,120/255,70/255))
    floor.SetLocalG(glm.scale(glm.vec3(20,0.01,20)))
    wall1=Objects(floor)
    wall1.SetColor(glm.vec3(1,1,1))
    wall1.SetLocalG(glm.scale(glm.vec3(20,5,0.01)))
    wall1.SetLocalM(glm.translate(glm.vec3(0,5,20)))
    wall2=Objects(floor)
    wall2.SetColor(glm.vec3(1,1,1))
    wall2.SetLocalG(glm.scale(glm.vec3(20,5,0.01)))
    wall2.SetLocalM(glm.translate(glm.vec3(0,5,-20)))
    wall3=Objects(floor)
    wall3.SetColor(glm.vec3(1,1,1))
    wall3.SetLocalG(glm.scale(glm.vec3(0.01,5,20)))
    wall3.SetLocalM(glm.translate(glm.vec3(20,5,0)))
    wall4=Objects(floor)
    wall4.SetColor(glm.vec3(1,1,1))
    wall4.SetLocalG(glm.scale(glm.vec3(0.01,5,20)))
    wall4.SetLocalM(glm.translate(glm.vec3(-20,5,0)))
    floor.SetGlobalM()

    character.pos=glm.vec3(10,0,10)
    character.tall=0
    character.SetLocalM(glm.translate(character.pos))
    character.n=5
    character.SetColor(glm.vec3(1,1,1))
    character.SetLocalG(glm.scale(glm.vec3(0.1,0.1,0.1)))
    character.SetGlobalM()
    
    while not glfwWindowShouldClose(window):
        timePerFrame=glfwGetTime()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glfwSetCursorPos(window,SCREEN_WIDTH/2,SCREEN_HEIGHT/2)

        #cam 업데이트
        if cam.clip:
            cam.gravityTimer+=GRAVITY_STEP
            cam.Pos.y+=-0.5*cam.gravityTimer**2
            if cam.Pos.y<=cam.tall:
                cam.Pos.y=cam.tall
                cam.gravityTimer=0

        cam.viewVector=glm.vec3(
            glm.cos(cam.angle1)*glm.cos(cam.angle2),
            glm.sin(cam.angle2),
            glm.sin(cam.angle1)*glm.cos(cam.angle2)
        )
        cam.Target=cam.Pos+cam.viewVector
        cam.armPos=cam.Pos+cam.viewVector*ARM_DISTANCE

        P=glm.perspective(cam.FOV,cam.aspectRatio,cam.near,cam.far)
        V=glm.lookAt(cam.Pos,cam.Target,cam.Up)
        M=glm.mat4()

        #object 위치업뎃
        for i,obj in enumerate(objects):
            if obj.n==cam.grabbedObject and cam.grab:
                obj.pos=cam.armPos
                obj.gravityTimer=0
                obj.angle=-cam.angle1-np.pi/2
            else:
                if obj.pos.y<=obj.tall:
                    obj.pos.y=obj.tall
                    obj.gravityTimer=0
                else:
                    obj.gravityTimer+=GRAVITY_STEP
                    obj.pos.y+=-0.5*obj.gravityTimer**2

        #objects 위치 업데이트 
        for obj in objects:
            T=glm.translate(obj.pos)
            R=glm.rotate(obj.angle,glm.vec3(0,1,0))
            obj.SetLocalM(T*R)
            obj.SetGlobalM()

        #ray체크
        ray=CheckRayCollision(cam.Pos,cam.viewVector,objects,1)
        if len(ray)==0:
            cam.shoot=False
        else:
            if cam.shoot:
                objects[ray[0]-1].SetColor(glm.vec3(1,0,0))
                cam.shoot=False


        #World Frame
        glUseProgram(shader)
        DrawFrame(vaoFrame,AmatLoc,P*V*M)

        #Cube 그리기
        for model in [floor,wall1,wall2,wall3,wall4]:
            Amat=P*V*model.GetModelMatrix()
            glUseProgram(shader_Object)
            DrawObject(vaoCube,AmatLocObject,Amat,
                       MmatLocObject,model.GetModelMatrix(),
                       vertexColorLocObject,model.GetColor(),
                       camVecLocObject,cam.Pos,
                       ambientLocObject,diffuseLocObject,
                       shininessLocObject,specularLocObject,
                       cubeModel)
        
        #sphere 그리기
        for sphere in spheres:
            Amat=P*V*sphere.GetModelMatrix()
            glUseProgram(shader_Object)
            DrawObject(vaoSphere,AmatLocObject,Amat,
                       MmatLocObject,sphere.GetModelMatrix(),
                       vertexColorLocObject,sphere.GetColor(),
                       camVecLocObject,cam.Pos,
                       ambientLocObject,diffuseLocObject,
                       shininessLocObject,specularLocObject,
                       sphereModel)
        
        #character 그리기
        Amat=P*V*character.GetModelMatrix()
        glUseProgram(shader_Object)
        DrawObject(vaoCharacter,AmatLocObject,Amat,
                   MmatLocObject,character.GetModelMatrix(),
                   vertexColorLocObject,character.GetColor(),
                   camVecLocObject,cam.Pos,
                   ambientLocObject,diffuseLocObject,
                   shininessLocObject,specularLocObject,
                   characterModel)
        
        glCullFace(GL_BACK)
        glfwSwapBuffers(window)
        glfwPollEvents()

        sleepTime=1/TARGET_FPS-glfwGetTime()+timePerFrame
        if sleepTime<0:
            sleepTime=0
        time.sleep(sleepTime)
        timePerFrame=glfwGetTime()-timePerFrame

        print(f"FPS: {1/timePerFrame:.2f}, Clip: {cam.clip}, Grab: {cam.grabbedObject}")


    glfwTerminate()


def CreateShader(vertexShaderSrc, fragmentShaderSrc):
    shader = compileProgram(
        compileShader( vertexShaderSrc, GL_VERTEX_SHADER ),
        compileShader( fragmentShaderSrc, GL_FRAGMENT_SHADER ),
    )
    return shader

def InitializeObject(model):
    dtype = np.dtype(np.float32).itemsize

    #vao
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    #vbo
    vbo = glGenBuffers(2)

    glBindBuffer(GL_ARRAY_BUFFER,vbo[0])
    glBufferData(GL_ARRAY_BUFFER,model.trivertex.nbytes,model.trivertex.flatten(),GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*dtype,None)

    glBindBuffer(GL_ARRAY_BUFFER,vbo[1])
    glBufferData(GL_ARRAY_BUFFER,model.trinormal.nbytes,model.trinormal.flatten(),GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,3*dtype,None)

    return vao

def DrawObject(vao,AmatLoc,Amat,MmatLoc,Mmat,vertexColorLoc,vertexColor,camVecLoc,camVec,ambientLoc,diffuseLoc,shininessLoc,specularLoc,model):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc,1,GL_FALSE,glm.value_ptr(Amat))
    glUniformMatrix4fv(MmatLoc,1,GL_FALSE,glm.value_ptr(Mmat))
    glUniform3fv(vertexColorLoc,1,glm.value_ptr(vertexColor))
    glUniform3fv(camVecLoc,1,glm.value_ptr(camVec))
    glUniform1f(ambientLoc,model.ambient)
    glUniform1f(diffuseLoc,model.diffuse)
    glUniform1f(shininessLoc,model.shininess)
    glUniform1f(specularLoc,model.specular)
    glDrawArrays(GL_TRIANGLES,0,model.polygons*3)

def InitializeWorldFrame():
    vertices = [0.,   0.,   0.,  
                10,    0.,   0.,  
                0.,   0.,   0.,  
                0.,   10.,   0.,  
                0.,   0.,   0.,  
                0.,   0.,   10.]  
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*9=36bytes

    colors = [  165./255., 42./255.,  42./255., # Brown
                165./255., 42./255.,  42./255., # Brown
                138./255., 43./255., 226./255., # Purple
                138./255., 43./255., 226./255., # Purple
                0.0,  1.0, 0.0, # Green
                0.0,  1.0, 0.0] # Green
    colors = np.array(colors, dtype=np.float32) # 4bytes*9=36bytes

    dtype = np.dtype(np.float32).itemsize

    # Create the name of Vertex Array Object(VAO)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)     

    # Generate two Vertex Buffer Object (VBO)
    vbo_ids = glGenBuffers(2) 
    # Bind the first VBO, copy the vertex data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)
    # Bind the second buffer, copy the color data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # the location of the vertex attribute: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    return vao

def DrawFrame(vao,AmatLoc,Amat):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    glDrawArrays(GL_LINES, 0, 6)

def isCollide(pos1,pos2):
    len=pos1-pos2
    if glm.length(len)<COLLIDE_THRESHOLD:
        return True
    return False

def CheckObjectsCollision(objs,pos):
    collisionList=[]
    nonCollisionList=[]
    for obj in objs:
        length=obj.pos-pos
        if glm.length(length)<COLLIDE_THRESHOLD:
            collisionList.append(obj.n)
        else:
            nonCollisionList.append(obj.n)
    return collisionList,nonCollisionList

def CheckRayCollision(rayPos,rayVec,objs,radius):
    collisionList=[]
    for obj in objs:
        pos=rayPos-obj.pos
        D=(glm.dot(rayVec,pos))**2-(glm.dot(rayVec,rayVec))*(glm.dot(pos,pos)-radius**2)
        if D>=0:
            collisionList.append(obj.n)
    return collisionList

vertexShaderSrc_object = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexNormal; 
out vec3 fragmentColor;
out vec3 viewpos;
out vec3 fragpos;
out vec3 normal;
out float ambientStrength;
out float specularStrength;
out float shininess;
out float diffuseStrength;

uniform mat4 Amat; 
uniform mat4 Mmat; 
uniform vec3 camVec;
uniform vec3 vertexColor;
uniform float ambientStrength_in;
uniform float specularStrength_in;
uniform float shininess_in;
uniform float diffuseStrength_in;


void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    
    // geometric information
    viewpos = camVec;
    fragpos = vec3(Mmat*point); // world (global) coordinates
    normal = normalize(inverse(transpose(mat3(Mmat)))*vertexNormal);

    // Material properties
    vec3 material_color = vertexColor;
    ambientStrength = ambientStrength_in;
    specularStrength = specularStrength_in;
    shininess = shininess_in;
    diffuseStrength = diffuseStrength_in;
    
    fragmentColor = material_color;

}
'''

fragmentShaderSrc_object = '''
#version 330 core
in vec3 viewpos;
in vec3 fragpos;
in vec3 normal;
in vec3 fragmentColor;
in float ambientStrength;
in float specularStrength;
in float shininess;
in float diffuseStrength;
out vec4 fragmentColorOut;

struct phLight { 
    vec3 color; 
    vec3 position; 
};

vec3 LightingModel(vec3 material_color, vec3 lightColor, 
        vec3 lightPosition, vec3 fragpos, vec3 normal, vec3 viewpos, 
        float ambientStrength_in, float specularStrength_in, float shininess_in, float diffuseStrength_in)
{
    //ambient
    vec3 ambient = ambientStrength_in*material_color*lightColor;

    // diffuse
    vec3 Lvec=normalize(lightPosition-fragpos);
    float cosTheta=max(dot(Lvec,normal),0.0);
    vec3 D=diffuseStrength_in*material_color;
    vec3 diffuse = D*cosTheta*lightColor;

    // specular
    vec3 Vvec=normalize(viewpos-fragpos);
    vec3 Rvec=2*dot(Lvec,normal)*normal-Lvec;
    float exponent=shininess_in;   //32
    float specFactor=pow(max(dot(Rvec,Vvec),0.0),exponent);
    vec3 specular = specFactor*specularStrength_in*lightColor;

    vec3 color =  ambient + diffuse + specular;
    return color;
}

void main()
{
    phLight light;
    light=phLight(vec3(1,1,1),vec3(0,8,0));
    vec3 color = LightingModel(fragmentColor,light.color,
                                light.position,fragpos,normal,viewpos,
                                ambientStrength,specularStrength,shininess,diffuseStrength);
    fragmentColorOut = vec4(color, 1.0);
}
'''

vertexShaderSrc_frame = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexColor; 
out vec3 fragmentColor;
uniform mat4 Amat; 
void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    fragmentColor = vertexColor;
}
'''
 


fragmentShaderSrc_frame = '''
#version 330 core
in vec3 fragmentColor;
out vec4 fragmentColorOut;
void main()
{
   fragmentColorOut = vec4(fragmentColor, 1.0); // alpha
}
'''

if __name__ == "__main__":
    main()