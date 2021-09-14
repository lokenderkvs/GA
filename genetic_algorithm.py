
                              #========== Name - Lokender Singh ======================================#
                              #========== Roll - 204103311      ======================================#
                              #========== Coding Assignment 2: Genetic Algorithm (Binary coded) ======#
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Minimize  (x1 + x2 - (2*x1*x1) - (x2*x2) + x1*x2)

# Objective function
def objective(x1,x2):
    return (1/(1+(x1 + x2 - (2*x1*x1) - (x2*x2) + x1*x2))) 
    
# Taking inputs
ps = int(input("Enter the Population Size: "))
cp = float(input("Enter the Crossover probability: "))
mp = float(input("Enter the mutation probability: "))
len_str1 = int(input("Enter the size of x1 in bits: "))
len_str2 = int(input("Enter the size of x2 in bits: "))
generation = int(input("Enter the number of generation you want to perform: "))

#Geometric constraints
x1min = 0
x1max = .5
x2min = 0
x2max = .5
pop = 2*ps

#======================================================== FUNCTIONS =====================================================================

# Decode values (x1,x2)
def decode(ps,X):
    d_val =np.zeros((ps))
    for j in range(ps):
        temp1 = X[j,:].tolist()
        temp2 = ''.join(str(i) for i in temp1)
        d_val[j] = int(temp2,2)
    return d_val

# Binary to decimal
def conversion(ps,xmin,xmax,d_val,len_str):
    x =np.zeros((ps,1))
    for i in range(ps):
        x[i] = xmin + ((xmax - xmin)/(2**len_str - 1))*d_val[i]
    return x

# fitness
def fitness(ps,x1,x2): 
    fit =np.zeros((ps))
    for i in range(ps):
        fit[i] = objective(x1[i],x2[i])
    return fit

# Maximum Fitness
mx = np.zeros(generation)
def maximum(fit):
    return np.max(fit)

# Minimum Fitness
mn = np.zeros(generation)
def minimum(fit):
    return np.min(fit)

# Average Fitness
av = np.zeros(generation)
def average(fit):
    return np.average(fit)

# Index of maximum fitness
def position(fit): 
    maxpos = np.argmax(fit)
    return maxpos

# value at maximum fitness
def solution(xmin,xmax,d_val,len_str,maxpos):
    z = xmin + ((xmax - xmin)/(2**len_str - 1))*d_val[maxpos]
    return z

# Roulette wheel
def Roulette(fit):
    c = np.cumsum(fit)
    r = sum(fit)*np.random.rand()
    index = np.argwhere(r<=c)
    return index[0][0]
    
def seq_matpool(ps,fit):
    seq =np.zeros((ps))
    for i in range(ps):
        seq[i] = Roulette(fit)
    seq = seq.astype(int)
    return seq

# Mating pool
def mating_pool(ps,len_string,seq):
    mat_pool = np.zeros(([ps,len_string]))
    for i in range(ps):
        a = seq[i]
        mat_pool[i,] = pop_matrix[a:a+1,:]
    return mat_pool

#Crossover (2 point)
def crossover(ps,ps_temp,len_string,cp):
    p1 =np.zeros(([ps_temp,len_string]))
    p2 =np.zeros(([ps_temp,len_string]))

    rand = random.sample(range(0,ps),ps)
    c=0
    for i in range(0,ps,2):
        a = rand[i]
        b = rand[i+1]
        p1[c,] = mat_pool[a:a+1,:]
        p2[c,] = mat_pool[b:b+1,:]
    
        r1 = random.uniform(0,1)
        if(r1<=cp):
            R = random.sample(range(1,len_string-1),2)
            R = sorted(R)
            tmp=0
            for k in range(R[0],R[1]+1):
                tmp = p1[c,k]
                p1[c,k]= p2[c,k]
                p2[c,k]= tmp
        c = c+1
    return p1,p2

#Mutation
def mutation(ps,ps_temp,len_string,p1,p2):

    for j in range(ps_temp):
        r2 = np.zeros((len_string))

        for k in range(len_string):
            r2[k] = random.random()

        for i in range(len_string):
            if(r2[i]<=mp):

                if(p1[j,i] == 1):
                    p1[j,i] = 0
                else:
                    p1[j,i] = 1

                if(p2[j,i] == 1):
                    p2[j,i] = 0
                else:
                    p2[j,i] = 1

    p3 = np.zeros(([ps,len_string]))
    c=0
    for i in range(0,ps_temp):
        p3[c,] = p1[i:i+1,:]
        p3[c+1,] = p2[i:i+1,:]
        c=c+2
    return p3

#=================================================  MAIN  =====================================================================

len_string = len_str1 + len_str2 
emp_arr = np.zeros([2*ps,len_string],int)

# Generation of population of solution
pop_matrix = np.random.randint(2,size=(ps,len_string))

iter=0
iter1 = np.zeros(generation)

A = np.zeros([ps,generation])
B = np.zeros([ps,generation])

while(iter<generation):
    # split initial population
    X1 = pop_matrix[:,0:len_str1]
    X2 = pop_matrix[:,len_str2:]

    # Decoded values (str1,str2)
    dval_1 = decode(ps,X1)
    dval_2 = decode(ps,X2)

    # Binary to decimal
    x1 = conversion(ps,x1min,x2max,dval_1,len_str1)
    x2 = conversion(ps,x1min,x2max,dval_2,len_str2)
    A[:,iter:iter+1] = x1
    B[:,iter:iter+1] = x2
        
    
    # Fitness
    fit = fitness(ps,x1,x2)

    #Maximum
    mx[iter] = maximum(fit)

    #Minimun
    mn[iter] = minimum(fit)

    #Average
    av[iter] = average(fit)

    #maxpos
    maxpos = position(fit)
    
    #solution
    z1=solution(x1min,x1max,dval_1,len_str1,maxpos)
    z2=solution(x2min,x2max,dval_2,len_str2,maxpos)
    
    #sequence of mating pool        
    seq = seq_matpool(ps,fit)

    #Mating Pool
    mat_pool = mating_pool(ps,len_string,seq)
                    
    #Crossover (2 point)
    ps_temp = int(0.5*ps)
    p1,p2 = crossover(ps,ps_temp,len_string,cp)            

    #Mutation
    pop_matrix1 = mutation(ps,ps_temp,len_string,p1,p2)
    pop_matrix1 = pop_matrix1.astype(int)
    
    #Survival
    emp_arr = np.concatenate((pop_matrix,pop_matrix1))

    Y1 = emp_arr[:,0:len_str1]
    Y2 = emp_arr[:,len_str2:]

    dval_y1 = decode(pop,Y1)
    dval_y2 = decode(pop,Y2)

    y1 = conversion(pop,x1min,x2max,dval_y1,len_str1)
    y2 = conversion(pop,x1min,x2max,dval_y2,len_str2)

    fit1 = fitness(pop,y1,y2)
    indices = (-fit1).argsort()[:ps]
    
    pop_matrix = np.zeros(([ps,len_string]))
    for i in range(ps):
        a = indices[i]
        pop_matrix[i,] = emp_arr[a:a+1, :]
    pop_matrix = pop_matrix.astype(int)
    
    iter1[iter]  = iter
    iter = iter+1

print("The value of x1 is = ",z1)
print("The value of x2 is = ",z2)


x = iter1

# Average Fitness vs No. of generations
plot1 = plt.figure(1)
plt.plot(x,av,color='b',label='Average Fitness')
plt.xlabel('No. of Generations')
plt.title('Avereage Fitness vs No. of Generations')
plt.legend()

# Maximum and Minimum Fitness vs No. of generations
plot2 = plt.figure(2)
plt.plot(x,mx,color='g',label='Maximum Fitness')
plt.plot(x,mn,color='r',label='Minimum Fitness')
plt.xlabel('No. of Generations')
plt.title('Maximum and Minimum Fitness vs No. of Generations')
plt.legend()

# Contour Plot for optimal solution
plt.style.use('default')
plt.figure(figsize=(12, 6))

x1 = np.linspace(0, 0.5, 22)
x2 = np.linspace(0,0.5, 22)

l, m = np.meshgrid(x1, x2)

z = l + m + - 2 * l ** 2 - m ** 2 + l * m

ax1 = plt.subplot(2, 3, 1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,0:1], B[:,0:1],label=' 1st generation')
plt.legend()
plt.colorbar()

ax2 = plt.subplot(232, sharey=ax1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,1:2], B[:,1:2],label=' 2nd generation')
plt.legend()
plt.colorbar()

ax3 = plt.subplot(233, sharey=ax1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,2:3], B[:,2:3],label=' 3rd generation')
plt.legend()
plt.colorbar()

plt.subplot(234, sharey=ax1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,3:4], B[:,3:4], label=' 4th generation')
plt.legend()
plt.colorbar()

plt.subplot(235, sharey=ax1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,4:5], B[:,4:5], label=' 5th generation')
plt.legend()
plt.colorbar()

plt.subplot(236, sharey=ax1)
plt.contour(l, m, z, 20)
plt.scatter(A[:,-1], B[:,-1], label=' Nth generation')
plt.legend()
plt.colorbar()

plt.show()
plt.close()

