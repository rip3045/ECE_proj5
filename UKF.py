# %%
from filterpy.kalman.sigma_points import SimplexSigmaPoints
from ECEproj4 import *
turn = 0.01/25
from numpy import matmul as matm
import filterpy
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import sqrtm

# %%
#check previous variables
#print(path.shape)    #the true path
#print(noise_path_intg.shape)    #the noisy model transition
#print(noised_path_sam1.shape)    #the noisy measurement data
#print(noisy_radar_bearing1.shape)
#print(noisy_radar_bearing2.shape)

noisy_radar1= np.zeros((200,3))
for i in range(0,200):
    noisy_radar1[i,:] = noisy_radar_bearing1[50*i,:]
    
print(noisy_radar1.shape)

noisy_radar2= np.zeros((200,3))

for i in range(0,200):
    noisy_radar2[i,:] = noisy_radar_bearing2[50*i,:]
    
print(noisy_radar2.shape)

# %%
###integrate over the noisy radar bearing


# %%
#use the f function for discrete nonlinear time, incorporates augmented vector
def f_forward(x_k1k1):
    x_kk1 = np.zeros((8,1))
    theta = x_k1k1[2]
    x_kk1[2] = theta+x_k1k1[4]
    x_kk1[0] = x_k1k1[0]+((0.5+x_k1k1[3])*np.cos(x_kk1[2]))
    x_kk1[1] = x_k1k1[1]+((0.5+x_k1k1[3])*np.sin(x_kk1[2]))
    x_kk1[2:8] = x_k1k1[2:8].reshape(6,1) 
    return x_kk1;
    

# %%
###define the measurement function for a posteriori calculation
def h_meas(x_kk1, radar_set):
    beta1 = np.arctan2((x_kk1[1]-radar_set1[0,1]),(x_kk1[0]-radar_set1[0,0]))
    beta2 = np.arctan2((x_kk1[1]-radar_set1[1,1]),(x_kk1[0]-radar_set1[1,0]))
    phi = x_kk1[2]
    y_k = np.zeros((8,1))
    y_k[0] = beta1+x_kk1[5]
    y_k[1] = beta2+x_kk1[6]
    y_k[2] = phi+x_kk1[7]
    y_k[2:8] = x_kk1[2:8].reshape(6,1) 
    return y_k;


# %%
#set up the tuning parameters 
nx = 3
nw = 2
nv = 3
#n_sig = 1+nx+nw+nv
n_sig=nx
filter_beta = 2
filter_alpha = 0.3
filter_kappa = 0
filter_lambda = (filter_alpha**2)*(n_sig+filter_kappa)-n_sig
w_0m = filter_lambda/(n_sig+filter_lambda)
w_0c = (filter_lambda/(n_sig+filter_lambda))+(1-(filter_alpha**2)+filter_beta)
w_jcm = 0.5/(n_sig+filter_lambda)
#filter_lambda = (filter_alpha**2)*(nx+filter_kappa)-nx
#w_0m = filter_lambda/(nx+filter_lambda)
#w_0c = (filter_lambda/(nx+filter_lambda))+(1-(filter_alpha**2)+filter_beta)
#w_jcm = 1/(nx+filter_lambda)
print("lambda: ",filter_lambda)
print("denominator: ", n_sig+filter_lambda)
print("mean first weight: ", w_0m)
print("covariance first weight: ", w_0c)
print("other weights: ", w_jcm)
print("total: ", w_0m+(16*w_jcm))

# %%
from filterpy.kalman import MerweScaledSigmaPoints
sigmas = MerweScaledSigmaPoints(3, alpha=.3, beta=2., kappa=0.)
print(sigmas)

# %%
##generates teh Merwe sigma points 
def gen_sig_points(x_aug, P_aug, filter_lambda, n_sig):
    ###generates a matrix of sigma points
    ##step to compute the contours
    #P_aug[0:3, 0:3] = 0.5*(P_aug[0:3, 0:3]+P_aug[0:3, 0:3].T)
    eig = np.linalg.eig(P_aug[0:3, 0:3])
    #print("eigenvalues: ", eig)
    add = sqrtm((filter_lambda+n_sig)*P_aug)
    #print(add)
    sig_mat = np.zeros((8,17))
    #print(sig_mat)
    for i in range(0,17):
        #print(sig_mat[:,i])
        sig_mat[:,i] = x_aug.reshape(8)
    sig_mat[:,1:9] = sig_mat[:,1:9]+add
    sig_mat[:,9:17] = sig_mat[:,9:17]-add
    return sig_mat;

# %%
#computes the next state given a sigma point
def forward_mat(sig_mat):
    x_kk1_mat = np.zeros((8,17))
    for i in range(0,17):
        x_kk1_mat[:,i] = f_forward(sig_mat[:,i]).reshape(8)
    return x_kk1_mat;

# %%
#compute the mean using a matrix of vectors and the weights
def compute_mean(vec_mat, w_0m, w_jcm):
    mean = w_0m*vec_mat[:,0]
    #print(mean)
    for i in range(1,17):
        mean = mean+(w_jcm*vec_mat[:,i])
        #print(x_mean)
    return mean/34;

# %%
#return the measurement vectors
def measure_mat(x_kk1_mat, radar_set):
    y_k_mat = np.zeros((8,17))
    for i in range(0,17):
        y_k_mat[:,i] = h_meas(x_kk1_mat[:,i], radar_set).reshape(8)
    return y_k_mat;

# %%
##computes a covariance matrix
def compute_covariance(vec_mat, mean, w_0c, w_jcm):
    mean = mean[0:3].reshape(3,1)
    e_vec = vec_mat[0:3,0].reshape(3,1)-mean
    P_cov = w_0c*(e_vec*e_vec.T)
    for i in range(1,17):
        e_vec = vec_mat[0:3,i].reshape(3,1)-mean
        P_cov = P_cov+(w_jcm*(e_vec*e_vec.T))
    return P_cov/34;


# %%
##computes the cross covariance matrix
def compute_cross_covariance(vec_mat1, mean1, vec_mat2, mean2, w_0c, w_jcm):
    mean1 = mean1[0:3].reshape(3,1)
    mean2 = mean2[0:3].reshape(3,1)
    e_vec1 = vec_mat1[0:3,0].reshape(3,1)-mean1
    e_vec2 = vec_mat2[0:3,0].reshape(3,1)-mean2
    P_c_cov = w_0c*(e_vec1*e_vec2.T)
    for i in range(1,17):
        e_vec1 = vec_mat1[0:3,i].reshape(3,1)-mean1
        e_vec2 = vec_mat2[0:3,i].reshape(3,1)-mean2
        P_c_cov = P_c_cov+(w_jcm*(e_vec1*e_vec2.T))
    return P_c_cov/34;

# %%
## one step in time, returns the pkk and xkk
def k_time_step(x_aug, P_aug, filter_lambda, nx, w_0m, w_0c, w_jcm, y_meas):

    sig_mat = gen_sig_points(x_aug, P_aug, filter_lambda, nx)
    x_kk1_mat = forward_mat(sig_mat)
    x_mean_priori = compute_mean(x_kk1_mat, w_0m, w_jcm)/17
    y_k_mat = measure_mat(x_kk1_mat, radar_set1)
    y_mean = compute_mean(y_k_mat, w_0m, w_jcm)/17
    P_x = compute_covariance(x_kk1_mat, x_mean_priori, w_0c, w_jcm)
    P_y = compute_covariance(y_k_mat, y_mean, w_0c, w_jcm)
    P_xy = compute_cross_covariance(x_kk1_mat, x_mean_priori, y_k_mat, y_mean, w_0c, w_jcm)
    Kk = P_xy@(np.linalg.inv(P_y))
    x_mean_posteriori = x_mean_priori[0:3].reshape(3,1)+Kk@(y_meas.reshape(3,1)+y_mean[0:3].reshape(3,1))
    #print("Kk shape: ", Kk.shape)
    #print("x priori shape: ", x_mean_priori.shape)
    #print(x_mean_posteriori)
    Pkk = P_x-(Kk@P_y@Kk.T)
    
    return x_mean_posteriori, Pkk;

# %%
##used to get a measure of the covariance
def m_std(P):
    sum = (P[0,0]**2)+(P[1,1]**2)+(P[2,2]**2)
    return np.sqrt(sum);

# %%
x_ukf = np.zeros((200,3))

# %%
x_ukf[0,:] = path[0,:]
x_ukf[0,:]
p_measured = np.zeros((200))

# %%
###set up the augmented covariance matrix
P_aug = np.zeros((8,8))
P_aug[0:3,0:3] = 0.001*np.eye(3)
Q = 2*np.eye(2)
Q[0,0] = 0.25
Q[1,1] = 0.01
R = 3*np.eye(3)
R[0,0] = deg2rad(3)**2
R[1,1] = R[0,0]
R[2,2] = deg2rad(np.sqrt(5))**2
P_aug[5:8, 5:8] = R
P_aug[3:5, 3:5] = Q
P_aug

# %%
#generate the data for 200 timesteps
x_k1k1 = x_ukf[0,:].reshape(3,1)
count = 0
for i in range(1,200):
    
    ##assign the augmented vector
    x_aug = np.zeros((8,1))
    x_aug[0:3] = x_k1k1
    #x_aug[0:3] = noise_path_intg[i,:].reshape(3,1)
    ##grab the sensor data
    y_meas = noisy_radar1[i,:]
    
    ##take one time step
    x_kk, P_kk = k_time_step(x_aug, P_aug, filter_lambda, nx, w_0m, w_0c, w_jcm, y_meas)
    
    ##document states
    x_ukf[i,:] = x_kk[0:3].reshape(3)
    p_measured[i] = m_std(P_kk)
    
    ##update vector and covariance for the next timestep
    x_k1k1 = x_kk
    P_aug[0:3, 0:3] = P_kk
    #print(P_aug)
    
    count +=1
    #print("successful loops: ", count)

# %%
print(x_ukf[:,0].shape)
print(x_ukf[0,:])
np.sin(x_ukf[0,2])

# %%
# plt.figure(figsize=(14,14))  
# plt.plot(Wptz[0].x,Wptz[0].y,'kx')
# plt.plot(Wptz[1].x,Wptz[1].y,'kx')
# plt.plot(path[:,0],path[:,1],'b-')    
# plt.plot(x_ukf[:,0], x_ukf[:,1], color='g')
# plt.scatter(noised_path_sam1[:,0], noised_path_sam1[:,1], color='r')
# #plt.plot(Ekalman_path_intg[:,0], Ekalman_path_intg[:,1], color='c')
# #plt.plot(x_Ekalman[:,0], x_Ekalman[:,1], color='c')

# plt.grid(True)
# plt.axis("equal")
# plt.title('Dubin\'s Curves With Kalman integral', fontsize='30')
# plt.xlabel('X', fontsize='30')
# plt.ylabel('Y', fontsize='30')
# plt.legend(['waypoint1', 'waypoint2','true path', 'kalman path','sensor estimate'])
# plt.show()

# %%
# plt.plot(p_measured, color='r')
# #plt.plot(Ekalman_path_intg[:,0], Ekalman_path_intg[:,1], color='c')
# #plt.plot(x_Ekalman[:,0], x_Ekalman[:,1], color='c')

# plt.grid(True)
# plt.axis("equal")
# plt.title('Dubin\'s Curves With Kalman integral', fontsize='30')
# plt.xlabel('X', fontsize='30')
# plt.ylabel('Y', fontsize='30')
# plt.legend(['p_sqrt'])
# plt.show()

# %%
x_ukf = np.zeros((200,3))
x_ukf[0,:] = path[0,:]
x_ukf[0,:]
p_measured = np.zeros((200))

# %%
P_aug = np.zeros((8,8))
P_aug[0:3,0:3] = 0.001*np.eye(3)
Q = 2*np.eye(2)
Q[0,0] = 0.25
Q[1,1] = 0.01
R = 3*np.eye(3)
R[0,0] = deg2rad(3)**2
R[1,1] = R[0,0]
R[2,2] = deg2rad(np.sqrt(5))**2
P_aug[5:8, 5:8] = R
P_aug[3:5, 3:5] = Q
P_aug


def angle_mean(sigmas, sp, angle_ind=(2, )):
    x = np.zeros(sigmas.shape[1])
    for i in angle_ind:
        sin_sum = np.sum(np.dot(np.sin(sigmas[:, i]), sp.Wm))
        cos_sum = np.sum(np.dot(np.cos(sigmas[:, i]), sp.Wm))
        x[i] = math.atan2(sin_sum, cos_sum)
    for i in range(sigmas.shape[1]):
        if i not in angle_ind:
            x[i] = np.sum(np.dot(sigmas[:, i], sp.Wm))
    return x


def unscented_transform(sigma, sp, angle_ind, noise_cov):

    # x = np.dot(self.sp.Wm, sigma)
    x = angle_mean(sigma, sp, angle_ind)
    # np.newaxis is a clever way to add a dimension
    y = sigma - x[np.newaxis, :]

    # normalize angles
    for i in angle_ind:
        y[:, i] = [normalize_radians(_y) for _y in y[:, i]]

    P = np.dot(y.T, np.dot(np.diag(sp.Wc), y))

    # add noise if it exists
    P = P + noise_cov if noise_cov is not None else P

    return x, P


def cross_variance(x, z, sp, sigmas_f, sigmas_h):

    Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))

    for i in range(sigmas_f.shape[0]):
        Pxz += sp.Wc[i] * np.outer(
            np.subtract(sigmas_f[i], x),
            np.subtract(sigmas_h[i], z)
        )

    return Pxz



def func_2_protect_locals():

    from scipy.stats import multivariate_normal

    RAD_2_DEGREE = 180 / np.pi

    # Max's Attempt
    R = np.diag([9 / (RAD_2_DEGREE ** 2), 9 /
                (RAD_2_DEGREE ** 2), 5 / (RAD_2_DEGREE ** 2)])

    Q = np.diag([0.05, 0.05, (1 / 5) ** 2 * 0.5 ** 2])

    R_func = multivariate_normal(mean=np.zeros(R.shape[0]), cov=R)

    local_x = None
    x = np.zeros_like(noise_path_intg)
    x[0] = path[0].copy()
    sigmas = MerweScaledSigmaPoints(3, alpha=10, beta=2., kappa=0.)
    posteriors = np.zeros((path.shape[0], path.shape[1], path.shape[1]))
    posteriors[0, :, :] = np.diag([0.1, ] * path.shape[1])
    prioris = posteriors.copy()

    ### Redoing the noisy measurements cause idk how u did it
    measurements = np.zeros_like(noise_path_intg)
    for i, s in enumerate(noise_path_intg):
        measurements[i, :] = np.array([
                    math.atan2((s[1] - -10), (s[0] - -15)),
                    math.atan2((s[1] - 5), (s[0] - -15)),
                    normalize_radians(s[-1])
                ])
        
        # measurements[i, :] += R_func.rvs()

    for i, measurement in enumerate(measurements):
        if i > len(noisy_radar2) - 2:
            break

        # predict
        local_x = x[i, :].copy()
        sigma = sigmas.sigma_points(local_x, posteriors[i])
        sigma_f = np.zeros_like(sigma)
        for k, s in enumerate(sigma):
            sigma_f[k, :] = np.array([
                    s[0] + 0.5 * 1 * np.cos(s[-1]),
                    s[1] + 0.5 * 1 * np.sin(s[-1]),
                    s[-1]
            ])
            sigma_f[k, 2] = normalize_radians(sigma_f[k, 2])
        
        _x, P_priori = unscented_transform(sigma_f, sigmas, angle_ind=(2, ), noise_cov=Q)

        sigmas_f = sigmas.sigma_points(_x, P_priori)

        prioris[i] = P_priori

        # update, use the actual measurement here
        sigmas_h = np.zeros_like(sigmas_f)
        for k, s in enumerate(sigmas_f):
            sigmas_h[k, :] =  np.array([
                math.atan2((s[1] - -10), (s[0] - -15)),
                math.atan2((s[1] - 5), (s[0] - -15)),
                s[-1]
            ])
            
        zp, S = unscented_transform(sigmas_h, sigmas, angle_ind=(0, 1, 2), noise_cov=R)

        Pxz = cross_variance(_x, zp, sigmas, sigmas_f, sigmas_h)

        K = np.dot(Pxz, np.linalg.inv(S))

        z_res = np.subtract(measurement, zp)
        z_res = np.array([normalize_radians(_y) for _y in z_res])

        _x = np.add(_x, np.dot(K, z_res))
        
        posteriors[i + 1] = P_priori - np.dot(K, np.dot(S, K.T))
        x[i + 1] = _x
    return x


x_ukf = func_2_protect_locals()

# %%
#generate the data for 200 timesteps
# x_k1k1 = x_ukf[0,:].reshape(3,1)
# count = 0
# for i in range(1,200):
    
#     ##assign the augmented vector
#     x_aug = np.zeros((8,1))
#     x_aug[0:3] = x_k1k1
#     #x_aug[0:3] = noise_path_intg[i,:].reshape(3,1)
#     ##grab the sensor data
#     y_meas = noisy_radar2[i,:]
    
#     ##take one time step
#     x_kk, P_kk = k_time_step(x_aug, P_aug, filter_lambda, nx, w_0m, w_0c, w_jcm, y_meas)
    
#     ##document states
#     x_ukf[i,:] = x_kk[0:3].reshape(3)
#     p_measured[i] = m_std(P_kk)
    
#     ##update vector and covariance for the next timestep
#     x_k1k1 = x_kk
#     P_aug[0:3, 0:3] = P_kk
#     #print(P_aug)
    
#     count +=1
#     #print("successful loops: ", count)

# # %%
# #clear the outliers in the inverted measurement data
# for i in range(0,200):
#     for j in range(0,2):
#         if np.abs(noised_path_sam2[i,j])>80:
#             noised_path_sam2[i,j] = noised_path_sam2[i-1,j]

# %%
plt.figure(figsize=(14,14))  
plt.plot(Wptz[0].x,Wptz[0].y,'kx')
plt.plot(Wptz[1].x,Wptz[1].y,'kx')
plt.plot(path[:,0],path[:,1],'b-')    
plt.plot(x_ukf[:,0], x_ukf[:,1], color='g')
plt.scatter(noise_path_intg[:,0], noise_path_intg[:,1], color='r')
#plt.plot(Ekalman_path_intg[:,0], Ekalman_path_intg[:,1], color='c')
#plt.plot(x_Ekalman[:,0], x_Ekalman[:,1], color='c')

plt.grid(True)
plt.axis("equal")
plt.title('Dubin\'s Curves With Kalman integral', fontsize='30')
plt.xlabel('X', fontsize='30')
plt.ylabel('Y', fontsize='30')
plt.legend(['waypoint1', 'waypoint2','true path', 'kalman path','sensor estimate'])
plt.show()

# %%



