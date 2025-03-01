import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def affine_transformation_ls(comparator_coords, reseau_coords):
    comparator_coords = np.array(comparator_coords)
    reseau_coords = np.array(reseau_coords)
    
    # set up design matrix A and observation vector l
    A = np.zeros((2 * len(comparator_coords), 6))
    l = np.zeros(2 * len(comparator_coords))
    
    # fill A and l
    for i, (comp, reseau) in enumerate(zip(comparator_coords, reseau_coords)):
        A[2*i, 0] = comp[0]  # a
        A[2*i, 1] = comp[1]  # b
        A[2*i, 4] = 1        # dx
        A[2*i, 5] = 0        # dy
        A[2*i+1, 2] = comp[0]  # c
        A[2*i+1, 3] = comp[1]  # d
        A[2*i+1, 4] = 0        # dx
        A[2*i+1, 5] = 1        # dy
        
        l[2*i] = reseau[0]
        l[2*i+1] = reseau[1]
        
    A_T_A = A.T @ A
    A_T_l = A.T @ l
    A_T_A_inv = np.linalg.inv(A_T_A)
        
    params = A_T_A_inv @ A_T_l
    a, b, c, d, dx, dy = params
    params_names = ['a', 'b', 'c', 'd', 'dx', 'dy']
    result_df = pd.DataFrame(params, index = params_names, columns=['Value'])
    
    residuals = A @ params - l
    residuals_x = []
    residuals_y = []
    for i in range(len(comparator_coords)):
        x = residuals[2*i]
        y = residuals[2*i+1]
        
        residuals_x.append(x)
        residuals_y.append(y)
    
    residuals_x = np.array(residuals_x)
    residuals_y = np.array(residuals_y)
    RMSE_x = np.sqrt(np.mean(residuals_x ** 2))
    RMSE_y = np.sqrt(np.mean(residuals_y ** 2))
    
    theta = np.degrees(np.arctan(c/a))
    Sx = np.sqrt(a**2 + c**2)
    Sy = np.sqrt(b**2 + d**2)
    delta = np.arctan((a*b + c*d)/(a*d - c*d))
    rotation_df = pd.DataFrame([theta, Sx, Sy, delta], index = ['𝜃', 'Sx', 'Sy', '𝛿'], columns=['Value'])
    
    print('Residuals:', residuals)
    print('RMSE_x:', RMSE_x)
    print('RMSE_y:', RMSE_y)
    print(result_df)
    print(rotation_df)
    return params, residuals_x, residuals_y

def residual_plot(reseau_coords, residuals_x, residuals_y, scale_factor=1000, point_size=15):
    reseau_coords = np.array(reseau_coords)
    
    residuals_x = residuals_x * scale_factor
    residuals_y = residuals_y * scale_factor
    
    fig, ax = plt.subplots(figsize = (8,8))
    for i in range(len(reseau_coords)):
        ax.plot([reseau_coords[i, 0], reseau_coords[i, 0] + residuals_x[i]], 
                [reseau_coords[i, 1], reseau_coords[i, 1] + residuals_y[i]], 
                color='black')
        
    ax.scatter(reseau_coords[:, 0], reseau_coords[:, 1], color='black', s=point_size)
    ax.set_xlim(-150,150)
    ax.set_ylim(-150,150)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Image Point Residual Plot')
    ax.set_aspect('equal')
    
    plt.show()

def to_fiducial_system(params, points):
    a, b, c, d, dx, dy = params
    points = np.array(points)
    transformed_points = np.array([
        [a * x + b * y + dx, c * x + d * y + dy]  # Affine transformation for each point
        for x, y in points
    ])
    
    return transformed_points

def systematic_error(transformed_points, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None):
    x_coords = transformed_points[:, 0]
    y_coords = transformed_points[:, 1]
    
    # minus offset
    x_adjusted = x_coords - principal_offset[0]
    y_adjusted = y_coords - principal_offset[1]
    
    # add radial distortion
    k0, k1, k2, k3 = radial_distortion
    r = np.sqrt(x_adjusted**2 + y_adjusted**2)
    x_rad = -x_adjusted * (k0 + k1 * r**2 + k2 * r**4 +k3 * r**6)
    y_rad = -y_adjusted * (k0 + k1 * r**2 + k2 * r**4 +k3 * r**6)
    
    # add decentering distortion
    p1, p2 = decentering_distortion
    x_dec = p1 * (r**2 + 2 * x_adjusted**2) + 2 * p2 * x_adjusted * y_adjusted
    y_dec = p2 * (r**2 + 2 * y_adjusted**2) + 2 * p1 * x_adjusted * y_adjusted
    
    # add atmospheric refraction
    if atmospheric_refraction is not None:
        H, h = atmospheric_refraction
        c = 153.358
        K = 2410 * H / (H**2 - 6 * H + 250) - 2410 * h / (h**2 - 6 * h + 250) * (h / H)
        x_atm = x_adjusted * K * (1 + r**2 / c**2)
        y_atm = y_adjusted * K * (1 + r**2 / c**2)
    else:
        # If atmospheric_refraction is not provided, skip atmospheric refraction correction
        x_atm = 0
        y_atm = 0
    
    # correct coordinates
    x_correct = x_adjusted + x_rad + x_dec + x_atm
    y_correct = y_adjusted + y_rad + y_dec + y_atm
    
    correct_points = np.column_stack((x_correct, y_correct))
    
    return correct_points
    



# part c
# fiducial points
fiducial_digital_27 = [
    [1345.25, -19285.75],
    [19162.25,-1468.75],
    [1344.5,-1469.75],
    [19163.5,-19281.75],
    [842,-10378],
    [19666,-10376.5],
    [10253,-966.5],
    [10253.25,-19787.75]
] 

fiducial_digital_28 = [
    [1345.25,-19285],
    [19173.75,-1482.75],
    [1357,-1471],
    [19162.5,-19296.25],
    [848,-10378],
    [19672.25,-10390],
    [10267.25,-972],
    [10253.25,-19794.5]
]

fiducial_true = [
    [-105.997,-105.995],
    [106.004,106.008],
    [-106.000,106.009],
    [106.012,-105.995],
    [-112.000,0.007],
    [112.006,0.007],
    [0.005,112.007],
    [0.002,-111.998]
] 

# tie points
tie_digital_27 = [
    [9112.75,-10878.25],
    [19254.5,-9756],
    [8524.75,-16100.25],
    [19133.75,-16965],
    [9192.25,-3894.5],
    [18587.5,-4037.75]
]

tie_digital_28 = [
    [1346.75,-10810],
    [11233.5,-9382.25],
    [912.5,-16032.25],
    [11398.5,-16602.25],
    [1197,-3730.5],
    [10654,-3575]
]

# Control points
control_digital_27 = [
    [10059,-10881.75],
    [11844,-17251.25],
    [17841.25,-18026.25],
    [14685.75,-18204.75],
    [11779.75,-1174]
]

control_digital_28 = [
    [2275.25,-10786],
    [4159.5,-17082.75],
    [10136.5,-17687.75],
    [6984,-17948.75],
    [3724.75,-853.25]
]

# Check points
check_digital_27 = [
    [9612.25,-14502.25],
    [14006.25,-9748.5],
    [9460.25,-2291.5]
]

check_digital_28 = [
    [1949.5,-14416.75],
    [6158.75,-9527.5],
    [1411,-2079.75]
]

# get params with image 27
result_27 = affine_transformation_ls(fiducial_digital_27, fiducial_true)
params1, residual_x_27, residual_y_27 = result_27
print(result_27)

# get residual plot of image 27
residual_plot_27 = residual_plot(fiducial_true, residual_x_27, residual_y_27)
print(residual_plot_27)

# convert tie, control and check points to fiducial system
tie_true_27 = to_fiducial_system(params1, tie_digital_27)
control_true_27 = to_fiducial_system(params1, control_digital_27)
check_true_27 = to_fiducial_system(params1, check_digital_27)
fiducial_true_27 = to_fiducial_system(params1, fiducial_digital_27)
print('Tie points in fiducial system:', tie_true_27)
print('Control points in fiducial system:', control_true_27)
print('Check points in fiducial system:', check_true_27)
print('fiducial points in image 27', fiducial_true_27)

# get params with image 28
result_28 = affine_transformation_ls(fiducial_digital_28, fiducial_true)
params2, residual_x_28, residual_y_28 = result_28
print(result_28)

# get residual plot of image 28
residual_plot_28 = residual_plot(fiducial_true, residual_x_28, residual_y_28)
print(residual_plot_28)

# convert tie, control and check points to fiducial system
tie_true_28 = to_fiducial_system(params2, tie_digital_28)
control_true_28 = to_fiducial_system(params2, control_digital_28)
check_true_28 = to_fiducial_system(params2, check_digital_28)
fiducial_true_28 = to_fiducial_system(params2, fiducial_digital_28)
print('Tie points in fiducial system:', tie_true_28)
print('Control points in fiducial system:', control_true_28)
print('Check points in fiducial system:', check_true_28)
print('fiducial points in image 28', fiducial_true_28)




# part d   
comparator_coords_check = [
    [-113.767, -107.4],
    [-43.717, -108.204],
    [36.361, -109.132],
    [106.408, -109.923],
    [107.189, -39.874],
    [37.137, -39.07],
    [-42.919, -38.158],
    [-102.968, -37.446],
    [-112.052, 42.714],
    [-42.005, 41.903],
    [38.051, 40.985],
    [108.089, 40.189],
    [108.884, 110.221],
    [38.846, 111.029],
    [-41.208, 111.961],
    [-111.249, 112.759]
]

reseau_coords_check = [
    [-110, -110],
    [-40, -110],
    [40, -110],
    [110, -110],
    [110, -40],
    [40, -40],
    [-40, -40],
    [-100, -40],
    [-110, 40],
    [-40, 40],
    [40, 40],
    [110, 40],
    [110, 110],
    [40, 110],
    [-40, 110],
    [-110, 110]
]

result2 = affine_transformation_ls(comparator_coords_check, reseau_coords_check)
print(result2)

plot = residual_plot(reseau_coords_check, result2[1], result2[2])
print(plot)



# part e
principal_offset = [
    -0.006, 0.006
]
radial_distortion = [
    0.8878E-04, -0.1528E-07, 0.5256E-12, 0.0000
]
decentering_distortion = [
    0.1346E-06, 0.1224E-07
]
atmospheric_refraction = [
    1844.408086, 1092.8025
]

tie_27_corrected = systematic_error(tie_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct tie points coords in image 27', tie_27_corrected)
control_27_corrected = systematic_error(control_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct control points coords in image 27', control_27_corrected)
check_27_corrected = systematic_error(check_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct check points coords in image 27', check_27_corrected)

tie_28_corrected = systematic_error(tie_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct tie points coords in image 28', tie_28_corrected)
control_28_corrected = systematic_error(control_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct control points coords in image 28', control_28_corrected)
check_28_corrected = systematic_error(check_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct check points coords in image 28', check_28_corrected)

fiducial_corrected = systematic_error(np.array(fiducial_true), principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None)
print('correct fiducial points coords using true values', fiducial_corrected)
fiducial_corrected = systematic_error(fiducial_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None)
print('correct fiducial points coords in image 27', fiducial_corrected)
fiducial_corrected = systematic_error(fiducial_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None)
print('correct fiducial points coords in image 28', fiducial_corrected)