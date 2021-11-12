from ECEproj4 import *
turn = 0.01/25
from numpy import matmul as matm

def init_P():
    P = np.eye(3)
    return P


## a priori state covariance
def Pkk1_cal(Fk1, Pk1k1, Lk1):
    Pkk1 = matm(Fk1,matm(Pk1k1,Fk1.T))+matm(Lk1,Lk1.T)
    return Pkk1


##innovation covariance
def Sk_cal(Hk, Pkk1, Mk):
    Sk = matm(matm(Hk,Pkk1),Hk.T)+matm(Mk,Mk.T)
    return Sk;


##sub-optimal kalman gain
def Kk_cal(Pkk1, Hk, Sk):
    Kk = matm(matm(Pkk1,Hk.T), np.linalg.inv(Sk))
    return Kk;


def h_meas(x_hatkk1, radar_set):
    beta1 = np.arctan2((x_hatkk1[1]-radar_set1[0,1]),(x_hatkk1[0]-radar_set1[0,0]))
    beta2 = np.arctan2((x_hatkk1[1]-radar_set1[1,1]),(x_hatkk1[0]-radar_set1[1,0]))
    phi = x_hatkk1[2]
    y = np.zeros((3,1))
    y[0] = beta1
    y[1] = beta2
    y[2] = phi
    return y;

    
##exact innovation
def y_tildek_cal(yk, x_hatkk1, radar_set):
    #print(yk)
    #print(h_meas(x_hatkk1, radar_set))
    y_tildek = yk-h_meas(x_hatkk1, radar_set)
    return y_tildek;


##a priori with innovation
def x_hatkk_cal(x_hatkk1, Kk, y_tildek):
    x_hatkk = x_hatkk1+matm(Kk,y_tildek)
    #print(matm(Kk,y_tildek))
    return x_hatkk;


#a posteriori state covariance
def Pkk_cal(Pkk1, Kk, Sk):
    Pkk = Pkk1-matm(Kk, matm(Sk,Kk.T))
    return Pkk;


def Hk_cal(x_kk1, radar_set):
    y_from_m = h_meas(x_kk1, radar_set)
    beta1 = y_from_m[0]
    beta2 = y_from_m[1]
    phi = x_kk1[2]
    y_k = x_kk1[1]
    x_k = x_kk1[0]
    denom1 = (1+beta1**2)*(xk-radar_set[0,0])
    denom2 = (1+beta2**2)*(xk-radar_set[1,0])
    Hk = np.eye(3)
    Hk[0,0] = -1*beta1/denom1
    Hk[0,1] = 1/denom1
    Hk[1,0] = -1*beta2/denom2
    Hk[1,1] = 1/denom2
    #Hk = np.array(([[-1*beta1/denom1, 1/denom1, 0], [-1*beta2/denom2, 1/denom2, 0], [0, 0, 1]]))
    return Hk;


def Fk_cal(x_kk1):
    theta = x_kk1[2]
    del_F = np.zeros((3,3))
    del_F[0,2] = np.cos(theta)
    del_F[1,2] = np.sin(theta)
    del_F[2,2] = 0
    return np.eye(3)+(0.01*del_F);


def Mk_cal():
    Mk = np.eye(3)
    Mk[0,0] = deg2rad(3)
    Mk[1,1] = deg2rad(3)
    Mk[2,2] = deg2rad(np.sqrt(5))
    return Mk;


def Lk_cal(x_kk1):
    theta = x_kk1[2]
    Lk = np.zeros((3,2))
    Lk[0,0] = 0.05*np.cos(theta)
    Lk[1,0] = 0.05*np.sin(theta)
    Lk[2,1] = 0.002
    return Lk;


def m_std(P):
    sum = (P[0,0]**2)+(P[1,1]**2)+(P[2,2]**2)
    return np.sqrt(sum);


Pkk = init_P()
Mk = Mk_cal()
Sk = np.eye(3)
x_hatk1k1 = noised_path1[0,:]
print(x_hatk1k1)


## initialize the measurement and the kalman path and the sk, pk stuff
p_post = np.zeros(path.shape[0])
p_post[0] = m_std(Pkk)
s_innov = np.zeros(path.shape[0])
s_innov[0] = m_std(Sk) 
x_kalman = np.zeros(path.shape)
y_innov = np.zeros(path.shape)
y_innov[0] = np.zeros(3)
x_kalman[0,:] = noise_path[0,:]


count = 0
for i in range(1,x_kalman.shape[0]):
    
    # this all 
    x_kk1 = noise_path[i,:].reshape(3,1)        #actual model measurement
    z = h_meas(noised_path1[i,:].reshape(3,1), radar_set1)          #actual radar measurement
    y_exp =  h_meas(x_kk1, radar_set1)
    y_exp = y_exp.reshape(3,1)
    yk = z-y_exp

    #calculate the matrices at this step
    Fk = Fk_cal(x_kk1)
    Hk = Hk_cal(x_kk1, radar_set1)
    Lk = Lk_cal(x_kk1)

    #kalman filter steps
    Pkk1 = Pkk1_cal(Fk, Pkk, Lk)
    Sk = Sk_cal(Hk, Pkk1, Mk)
    Kk = Kk_cal(Pkk1, Hk, Sk)
    y_tildek = y_tildek_cal(z, x_kk1, radar_set1)
    x_hatkk = x_hatkk_cal(x_kk1, Kk, y_tildek)
    Pkk = Pkk_cal(Pkk1, Kk, Sk)
    count +=1

    ## data recording
    x_kalman[i,:] = x_hatkk.reshape(3)
    y_innov[i,:] = y_tildek.reshape(3)
    p_post[i] = m_std(Pkk)
    s_innov[i] = m_std(Sk)
    #print("loop number", count)
        
    