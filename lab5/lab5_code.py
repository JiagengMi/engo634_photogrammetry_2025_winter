import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import inspect
from mpl_toolkits.mplot3d import Axes3D

# Lab2: To perform image point measurement, interior orientation and image point refinement for a pair of images as part of the overarching aim of labs 2-5 of photogrammetric restitution
print('Lab2: To perform image point measurement, interior orientation and image point refinement for a pair of images as part of the overarching aim of labs 2-5 of photogrammetric restitution')
# part c
# fiducial points
fiducial_digital_27 = pd.DataFrame(data=[
    [1345.25, -19285.75],
    [19162.25,-1468.75],
    [1344.5,-1469.75],
    [19163.5,-19281.75],
    [842,-10378],
    [19666,-10376.5],
    [10253,-966.5],
    [10253.25,-19787.75]
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

fiducial_digital_28 = pd.DataFrame(data=[
    [1345.25,-19285],
    [19173.75,-1482.75],
    [1357,-1471],
    [19162.5,-19296.25],
    [848,-10378],
    [19672.25,-10390],
    [10267.25,-972],
    [10253.25,-19794.5]
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

fiducial_true = pd.DataFrame(data=[
    [-105.997,-105.995],
    [106.004,106.008],
    [-106.000,106.009],
    [106.012,-105.995],
    [-112.000,0.007],
    [112.006,0.007],
    [0.005,112.007],
    [0.002,-111.998]
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

# tie points
tie_digital_27 = pd.DataFrame(data=[
    [9112.75,-10878.25],
    [19254.5,-9756],
    [8524.75,-16100.25],
    [19133.75,-16965],
    [9192.25,-3894.5],
    [18587.5,-4037.75]
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6'])

tie_digital_28 = pd.DataFrame(data=[
    [1346.75,-10810],
    [11233.5,-9382.25],
    [912.5,-16032.25],
    [11398.5,-16602.25],
    [1197,-3730.5],
    [10654,-3575]
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6'])

# Control points
control_digital_27 = pd.DataFrame(data=[
    [10059,-10881.75],
    [11844,-17251.25],
    [17841.25,-18026.25],
    [9612.25,-14502.25],
    [11779.75,-1174]
], columns=['x(mm)', 'y(mm)'], index=['102', '104', '105', '202', '200'])

control_digital_28 = pd.DataFrame(data=[
    [2275.25,-10786],
    [4159.5,-17082.75],
    [10136.5,-17687.75],
    [1949.5,-14416.75],
    [3724.75,-853.25]
], columns=['x(mm)', 'y(mm)'], index=['102', '104', '105', '202', '200'])

# Check points
check_digital_27 = pd.DataFrame(data=[
    [14685.75,-18204.75],
    [14006.25,-9748.5],
    [9460.25,-2291.5]
], columns=['x(mm)', 'y(mm)'], index=['203', '201', '100'])

check_digital_28 = pd.DataFrame(data=[
    [6984,-17948.75],
    [6158.75,-9527.5],
    [1411,-2079.75]
], columns=['x(mm)', 'y(mm)'], index=['203', '201', '100'])

# Estimate the linear parameters and derive the non-linear parameters for the 2D affine transformation
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

# Plot residuals
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
    
    # plt.show()

# Transform all measured points using the estimated transformation parameters
def to_fiducial_system(params, points):
    a, b, c, d, dx, dy = params
    points_array = np.array(points)
    transformed_points = np.array([
        [a * x + b * y + dx, c * x + d * y + dy]  # Affine transformation for each point
        for x, y in points_array
    ])
    transformed_points_df = pd.DataFrame(data=transformed_points, columns=['x(mm)', 'y(mm)'])
    transformed_points_df.index = points.index
    return transformed_points_df

# get params with image 27
result_27 = affine_transformation_ls(fiducial_digital_27, fiducial_true)
params1, residual_x_27, residual_y_27 = result_27
print(result_27)

# get residual plot of image 27
# residual_plot_27 = residual_plot(fiducial_true, residual_x_27, residual_y_27)
# print(residual_plot_27)

# convert tie, control and check points to fiducial system
tie_true_27 = to_fiducial_system(params1, tie_digital_27)
control_true_27 = to_fiducial_system(params1, control_digital_27)
check_true_27 = to_fiducial_system(params1, check_digital_27)
fiducial_true_27 = to_fiducial_system(params1, fiducial_digital_27)
print('Tie points in fiducial system:\n', tie_true_27)
print('Control points in fiducial system:\n', control_true_27)
print('Check points in fiducial system:\n', check_true_27)
print('fiducial points in image 27:\n', fiducial_true_27)

# get params with image 28
result_28 = affine_transformation_ls(fiducial_digital_28, fiducial_true)
params2, residual_x_28, residual_y_28 = result_28
print(result_28)

# get residual plot of image 28
# residual_plot_28 = residual_plot(fiducial_true, residual_x_28, residual_y_28)
# print(residual_plot_28)

# convert tie, control and check points to fiducial system
tie_true_28 = to_fiducial_system(params2, tie_digital_28)
control_true_28 = to_fiducial_system(params2, control_digital_28)
check_true_28 = to_fiducial_system(params2, check_digital_28)
fiducial_true_28 = to_fiducial_system(params2, fiducial_digital_28)
print('Tie points in fiducial system:\n', tie_true_28)
print('Control points in fiducial system:\n', control_true_28)
print('Check points in fiducial system:\n', check_true_28)
print('fiducial points in image 28:\n', fiducial_true_28)


# part d   
comparator_coords_check = pd.DataFrame(data=[
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
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6', '7' ,'8', '9', '10', '11', '12', '13', '14', '15', '16'])

reseau_coords_check = pd.DataFrame(data=[
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
], columns=['x(mm)', 'y(mm)'], index=['1', '2', '3', '4', '5', '6', '7' ,'8', '9', '10', '11', '12', '13', '14', '15', '16'])

result2 = affine_transformation_ls(comparator_coords_check, reseau_coords_check)
print(result2)

# plot = residual_plot(reseau_coords_check, result2[1], result2[2])
# print(plot)



# part e
# Provided information
principal_offset = pd.DataFrame(data=[[
    -0.006, 0.006
]], columns=['x', 'y'])
radial_distortion = pd.DataFrame(data=[[
    0.8878E-04, -0.1528E-07, 0.5256E-12, 0.0000
]], columns=['k0', 'k1', 'k2', 'k3'])
decentering_distortion = pd.DataFrame(data=[[
    0.1346E-06, 0.1224E-07
]], columns=['p1', 'p2'])
atmospheric_refraction = pd.DataFrame(data=[[
    1844.408086, 1092.8025
]], columns=['H', 'h'])

# Perform all of image point corrections
def systematic_error(transformed_points, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None):
    transformed_points_array = np.array(transformed_points)
    principal_offset = np.array(principal_offset).flatten()
    radial_distortion = radial_distortion.to_numpy().flatten()
    decentering_distortion = decentering_distortion.to_numpy().flatten()
    if atmospheric_refraction is not None:
        atmospheric_refraction = atmospheric_refraction.to_numpy().flatten()
    
    x_coords = transformed_points_array[:, 0]
    y_coords = transformed_points_array[:, 1]
    
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
        H= H/1000
        h= h/1000
        c = 153.358
        K = (2410 * H / (H**2 - 6 * H + 250) - 2410 * h / (h**2 - 6 * h + 250) * (h / H)) * 10e-6
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
    correct_points_df = pd.DataFrame(data=correct_points, columns=['x(mm)', 'y(mm)'])
    correct_points_df.index = transformed_points.index
    
    return correct_points_df
    

# Apply correction to each of point
tie_27_corrected = systematic_error(tie_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct tie points coords in image 27\n', tie_27_corrected)
control_27_corrected = systematic_error(control_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct control points coords in image 27\n', control_27_corrected)
check_27_corrected = systematic_error(check_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct check points coords in image 27\n', check_27_corrected)

tie_28_corrected = systematic_error(tie_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct tie points coords in image 28\n', tie_28_corrected)
control_28_corrected = systematic_error(control_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct control points coords in image 28\n', control_28_corrected)
check_28_corrected = systematic_error(check_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
print('correct check points coords in image 28\n', check_28_corrected)

fiducial_corrected = systematic_error(fiducial_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None)
print('correct fiducial points coords in image 27\n', fiducial_corrected)
fiducial_corrected = systematic_error(fiducial_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction=None)
print('correct fiducial points coords in image 28\n', fiducial_corrected)
























# Lab3: To perform relative orientation of a stereo pair and estimate 3D model coordinates as part of the overarching aim of labs 2-5 of photogrammetric restitution.
print('\n' * 5)
print('Lab3: To perform relative orientation of a stereo pair and estimate 3D model coordinates as part of the overarching aim of labs 2-5 of photogrammetric restitution.')
# Plot tie points's coordinates
def points_coordinates(points, number, point_size=15):
    fig, ax = plt.subplots(figsize = (8,8))
    points_arr = points[['X (mm)', 'Y (mm)']].values
    ax.scatter(points_arr[:, 0], points_arr[:, 1], color='black', s=point_size, edgecolors='black', facecolors='none')
    
    # Add labels with index and coordinates
    for i, (x, y) in enumerate(points_arr):
        ax.text(x, y, f'({x:.2f}, {y:.2f})\n{points.index[i]}', fontsize=9, ha='left', va='bottom')
    
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'{number} GCP points')
    ax.set_aspect('equal')
    
    # plt.savefig(f'F:\\04-UoC master\OneDrive - University of Calgary\\02-UoC\\25winter_ENGO634_Principal of Photogrammetry\lab_code\lab4\\{number}.png')
    plt.show()

    
# Get plots of tie points of two images
# tie_plot_27 = points_coordinates(tie_27_corrected, 27)
# print(tie_plot_27)

# tie_plot_28 = points_coordinates(tie_28_corrected, 28)
# print(tie_plot_28)

# Function to compute the rotation matrix from omega, phi, kappa
def compute_rotation_matrix(omega, phi, kappa):
    omega, phi, kappa = np.radians([omega, phi, kappa])  # Convert to radians
    R_omega = np.array([
        [1, 0, 0],
        [0, np.cos(omega), np.sin(omega)],
        [0, -np.sin(omega), np.cos(omega)]
    ])
    R_phi = np.array([
        [np.cos(phi), 0, -np.sin(phi)],
        [0, 1, 0],
        [np.sin(phi), 0, np.cos(phi)]
    ])
    R_kappa = np.array([
        [np.cos(kappa), np.sin(kappa), 0],
        [-np.sin(kappa), np.cos(kappa), 0],
        [0, 0, 1]
    ])
    return R_kappa @ R_phi @ R_omega  # Combined rotation matrix

# Calculate relative orientation parameters
def relative_orientation(left_points, right_points, bx, c, initial_guess):
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # Error function (Misclosure vector)
    def error_function(params):
        by, bz, omega, phi, kappa = params
        # Rotation matrix M for the current parameters
        M_0 = compute_rotation_matrix(omega, phi, kappa)  # Compute rotation matrix
        
        misclosure = []
        
        for i in range(len(left_points)):
            xL, yL = left_points[i]
            xR, yR = right_points[i]
            
            # Right vector (before rotation)
            right_vector = np.array([xR, yR, -c])
            
            # Apply rotation to right vector
            transformed_point = M_0.T @ right_vector + np.array([bx, by, bz])
            
            # Construct delta (Misclosure vector)
            delta_value = np.array([[bx, by, bz], [xL, yL, -c], [transformed_point[0], transformed_point[1], transformed_point[2]]])
            det = np.linalg.det(delta_value)
            misclosure.append(det)
        
        #misclosure = np.array(misclosure).flatten()  # Flatten to a single vector
        return misclosure
    
    # Use scipy's least_squares to minimize the error function
    result = least_squares(error_function, initial_guess)
    
    # The optimal parameters
    bY_best, bZ_best, omega_best, phi_best, kappa_best = result.x
    # omega_best_d, phi_best_d, kappa_best_d = np.degrees([omega_best, phi_best, kappa_best])
    
    # Put them into a pd frame
    params = [bY_best, bZ_best, omega_best, phi_best, kappa_best]
    df_params = pd.DataFrame([params], columns=['By', 'Bz', 'Omega', 'Phi', 'Kappa'])
    
    # (J^T J) and its inverse
    J = result.jac
    JTJ = J.T @ J
    JTJ_inv = np.linalg.inv(JTJ)
    C_x = JTJ_inv
    
    # Compute correlation matrix
    diag_std = np.sqrt(np.diag(C_x))
    correlation_matrix = C_x / np.outer(diag_std, diag_std)
    
    C_df = pd.DataFrame(correlation_matrix, columns=['By', 'Bz', 'Omega', 'Phi', 'Kappa'], index=['By', 'Bz', 'Omega', 'Phi', 'Kappa'])
    
    return df_params, C_df


def convert_to_RO(bx, c, df_params, left_points, right_points):
    # Compute rotation matrix
    bY_best, bZ_best, omega_best, phi_best, kappa_best = df_params.iloc[0,:]
    # omega_best, phi_best, kappa_best = np.deg2rad([omega_best, phi_best, kappa_best])
    M = compute_rotation_matrix(omega_best, phi_best, kappa_best)
    left_points_array = np.hstack([left_points, -c*np.ones((left_points.shape[0], 1))])
    right_points = np.hstack([right_points, -c*np.ones((right_points.shape[0], 1))])
    
    # Compute scale lambda and mu and get final coordinates
    scale_lambda = []
    scale_mu = []
    final_coords = []
    pY = []
    for i in range(len(left_points_array)):
        xL,yL,zL = left_points_array[i]
        xR,yR,zR = M.T @ right_points[i]
        
        scale_lambda_value = (bx*zR-bZ_best*xR)/(xL*zR+c*xR)
        scale_lambda.append(scale_lambda_value)
        
        scale_mu_value = (-bx*c-bZ_best*xL)/(xL*zR+c*xR)
        scale_mu.append(scale_mu_value)
        
        xLM = scale_lambda_value * xL
        yLM = scale_lambda_value * yL
        zLM = scale_lambda_value * zL
        
        xRM = scale_mu_value * xR + bx
        yRM = scale_mu_value * yR + bY_best
        zRM = scale_mu_value * zR + bZ_best
        
        pY_value = yRM - yLM
        pY.append(pY_value)
        
        final_coords.append([xLM, (yLM+yRM)/2, zLM])
    
    # put them into a dataframe
    df_points = pd.DataFrame({
        'xl (mm)': left_points_array[:, 0],
        'yl (mm)': left_points_array[:, 1],
        'xr (mm)': right_points[:, 0],
        'yr (mm)': right_points[:, 1],
        'pY (mm)': pY,
        'X (mm)': [coords[0] for coords in final_coords],
        'Y (mm)': [coords[1] for coords in final_coords],
        'Z (mm)': [coords[2] for coords in final_coords]
    })
    df_points.index = left_points.index
    
    return df_points
    


# Use tie points compute 5 unknowns
bx = 92
c = 153.358
initial_guess = np.array([0,0,0,0,0])

tie_params, tie_cov = relative_orientation(tie_27_corrected, tie_28_corrected, bx, c, initial_guess)
print(tie_params)
print("\n")
print(tie_cov)
tie_points = convert_to_RO(bx, c, tie_params, tie_27_corrected, tie_28_corrected)
print(tie_points)

# Convert control points and check points
check_points = convert_to_RO(bx, c, tie_params, check_27_corrected, check_28_corrected)
print(check_points)

control_points = convert_to_RO(bx, c, tie_params, control_27_corrected, control_28_corrected)
print(control_points)

# Use lecture notes to test program
left_test = pd.DataFrame([[106.399, 90.426],
                     [18.989,93.365],
                     [70.964,4.907],
                     [-0.931,-7.284],
                     [9.278,-92.926],
                     [98.681,-62.769]], index=['30', '40', '72', '127', '112', '50'], columns=['x(mm)', 'y(mm)'])
right_test = pd.DataFrame([[24.848,81.824],
                     [-59.653,88.138],
                     [-15.581,-0.387],
                     [-85.407,-8.351],
                     [-78.81,-92.62],
                     [8.492,-68.873]], index=['30', '40', '72', '127', '112', '50'], columns=['x(mm)', 'y(mm)'])
bx_test = 92
c_test = 152.15
initial_guess_test = np.array([0,0,0,0,0])

test_params, test_cov = relative_orientation(left_test, right_test, bx_test, c_test, initial_guess_test)

print(test_params)      
print("\n")  
print(test_cov)
test_points = convert_to_RO(bx_test, c_test, test_params, left_test, right_test) 
print(test_points)


























# Lab4: To perform absolute orientation and check point analysis as part of the overarching aim of labs 2-5 of photogrammetric restitution
print('\n' * 5)
print('Lab4: To perform absolute orientation and check point analysis as part of the overarching aim of labs 2-5 of photogrammetric restitution')
check_points_copy = check_points.copy()
control_points_copy = control_points.copy()
test_points_copy = test_points.copy()

check_points_copy[['X (m)', 'Y (m)', 'Z (m)']] = pd.DataFrame([[527.78, -375.72, 1092.00], 
                                                               [42.73, -412.19, 1090.82], 
                                                               [-399.28, -679.72, 1090.96]], 
                                                              index=['203', '201', '100'])

control_points_copy[['X (m)', 'Y (m)', 'Z (m)']] = pd.DataFrame([[109.70, -642.35, 1086.43], 
                                                                [475.55, -538.18, 1090.50], 
                                                                [517.62, -194.43, 1090.65], 
                                                                [321.09, -667.45, 1083.49], 
                                                                [-466.39, -542.31, 1091.55]], 
                                                               index=['102', '104', '105', '202', '200'])

test_points_copy[['X (m)', 'Y (m)', 'Z (m)']] = pd.DataFrame([[7350.27, 4382.54, 276.42], 
                                                              [6717.22, 4626.41, 280.05], 
                                                              [6869.09, 3844.56, 283.11], 
                                                              [6316.06, 3934.63, 283.03], 
                                                              [6172.84, 3269.45, 248.10], 
                                                              [6905.26, 3279.84, 266.47]], 
                                                             index=['30', '40', '72', '127', '112', '50'])

def plot_3d_points(points):
    x = points.iloc[:, 0]  
    y = points.iloc[:, 1]  
    z = points.iloc[:, 2]  

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def absolute_orientation(points):
    # Print the title
    frame = inspect.currentframe()
    caller_locals = frame.f_back.f_locals  # Get the local variables from the caller's frame
    for var_name, var_value in caller_locals.items():
        if var_value is points:  # Match the value passed with the name
            print(f"Results for {var_name}:\n")
            break
    
    # Get coordinates in model space and object space        
    points_arr = np.array(points)
    x_m = points_arr[:,5]
    y_m = points_arr[:,6]
    z_m = points_arr[:,7]
    x_o = points_arr[:,8]
    y_o = points_arr[:,9]
    z_o = points_arr[:,10]
    
    # Error function (Misclosure vector)
    def error_function(params):
        omega, phi, kappa, scale, tx, ty, tz = params
        # Rotation matrix M for the current parameters
        M = compute_rotation_matrix(omega, phi, kappa)  # Compute rotation matrix
        
        misclosure = []
        
        for i in range(len(points)):
            misclosure.append(scale * (M[0, 0] * x_m[i] + M[0, 1] * y_m[i] + M[0, 2] * z_m[i]) + tx - x_o[i])
            misclosure.append(scale * (M[1, 0] * x_m[i] + M[1, 1] * y_m[i] + M[1, 2] * z_m[i]) + ty - y_o[i])
            misclosure.append(scale * (M[2, 0] * x_m[i] + M[2, 1] * y_m[i] + M[2, 2] * z_m[i]) + tz - z_o[i])
    
        return misclosure

    # Get initial guess from points, choosing the first two
    a_o = np.rad2deg(np.atan((x_o[0]-x_o[1])/(y_o[0]-y_o[1])))
    a_m = np.rad2deg(np.atan((x_m[0]-x_m[1])/(y_m[0]-y_m[1])))
    kappa = a_o - a_m
    scale = np.sqrt((x_o[0]-x_o[1])**2 + (y_o[0]-y_o[1])**2 + (z_o[0]-z_o[1])**2) / np.sqrt((x_m[0]-x_m[1])**2 + (y_m[0]-y_m[1])**2 + (z_m[0]-z_m[1])**2)

    initial_guess = [0,0,kappa,scale,0,0,0]
    
    # Use scipy's least_squares to minimize the error function
    result = least_squares(error_function, initial_guess)
    
    # The optimal parameters
    omega, phi, kappa, scale, tx, ty, tz = result.x
    
    # Put them into a pd frame
    params = [omega, phi, kappa, scale, tx, ty, tz]
    df_params = pd.DataFrame([params], columns=['Ω (deg)', 'Φ (deg)', 'Κ (deg)', 'Λ', 'tx', 'ty', 'tz'])
    
    # (J^T J) and its inverse
    J = result.jac
    JTJ = J.T @ J
    JTJ_inv = np.linalg.inv(JTJ)
    C_x = JTJ_inv
    
    # Compute correlation matrix
    diag_std = np.sqrt(np.diag(C_x))
    correlation_matrix = C_x / np.outer(diag_std, diag_std)
    
    C_df = pd.DataFrame(correlation_matrix, columns=['Ω (deg)', 'Φ (deg)', 'Κ (deg)', 'Λ', 'tx', 'ty', 'tz'], index=['Ω (deg)', 'Φ (deg)', 'Κ (deg)', 'Λ', 'tx', 'ty', 'tz'])
    print('Correlation matrix:\n', C_df)
    
    # Calculate redundancy number
    u = 7
    n = len(points) * 3
    r = n - u
    print(f'The total redundancy number is: {r}')
    
    # Calculate redundancy number for each coord
    P = np.eye(n)
    I = np.eye(n)
    R = I - J @ C_x @ J.T @ P
    diagonal = [R[i, i] for i in range(n)]
    diagonal_reshape = np.array(diagonal).reshape(-1,3)
    df_R = pd.DataFrame(diagonal_reshape, index=points.index, columns=['x', 'y', 'z'])
    df_R.loc['Total'] = df_R.sum()
    print("Redundancy number for each coord:\n", df_R)
    
    # Check results using trace
    trace_R = np.trace(R)
    print("Check redundancy number using trace:", trace_R)
    
    # Plot GCPs
    # plot_points = points[['X (mm)', 'Y (mm)']]
    # plot_points_3d = points[['X (mm)', 'Y (mm)', 'Z (mm)']]
    # plot = points_coordinates(plot_points, len(points))
    # plot_3d = plot_3d_points(plot_points_3d)
    # print(plot)
    # print(plot_3d)
    return df_params

def residual(params, points):

    omega, phi, kappa, scale, tx, ty, tz = params.iloc[0]
    M = compute_rotation_matrix(omega, phi, kappa)
    
    points_arr = np.array(points)
    x_m = points_arr[:,5]
    y_m = points_arr[:,6]
    z_m = points_arr[:,7]
    x_o = points_arr[:,8]
    y_o = points_arr[:,9]
    z_o = points_arr[:,10]
    
    # Calculate residuals
    residuals_x = []
    residuals_y = []
    residuals_z = []
    
    for i in range(len(points)):
        residuals_x.append(scale * (M[0, 0] * x_m[i] + M[0, 1] * y_m[i] + M[0, 2] * z_m[i]) + tx - x_o[i])
        residuals_y.append(scale * (M[1, 0] * x_m[i] + M[1, 1] * y_m[i] + M[1, 2] * z_m[i]) + ty - y_o[i])
        residuals_z.append(scale * (M[2, 0] * x_m[i] + M[2, 1] * y_m[i] + M[2, 2] * z_m[i]) + tz - z_o[i])
    
    # Calculate mean and rms of errors
    residuals_x = np.array(residuals_x)  # Convert to numpy arrays
    residuals_y = np.array(residuals_y)
    residuals_z = np.array(residuals_z)
    
    Mean = np.array([np.mean(residuals_x), np.mean(residuals_y), np.mean(residuals_z)])
    RMSE = np.array([np.sqrt((residuals_x ** 2).mean()), np.sqrt((residuals_y ** 2).mean()), np.sqrt((residuals_z ** 2).mean())])
    
    residuals_df = pd.DataFrame({
        'x':residuals_x,
        'y':residuals_y,
        'z':residuals_z
    }, index=points.index)
    
    residuals_df.loc['Mean'] = [Mean[0], Mean[1], Mean[2]]
    residuals_df.loc['RMSE'] = [RMSE[0], RMSE[1], RMSE[2]]
    
    return residuals_df



# Results for test data
test_result = absolute_orientation(test_points_copy)
test_residual = residual(test_result, test_points_copy)
print(test_result)
print('Residuals for test data:\n', test_residual)

# Results for GCPs
# Just use 105, 202, 200
# control_1 = control_points_copy.loc[['105', '202', '200']]
# control_check = pd.concat([control_1, check_points_copy], axis = 0)
# control_result_1 = absolute_orientation(control_1)
# control_residual_1 = residual(control_result_1, control_check)
# check_control = pd.concat([control_points_copy.loc[['102', '104']], check_points_copy], axis = 0)
# control_residual_5check = residual(control_result_1, check_control)
# print(control_result_1)
# print('Residuals for 3 GCPs data:\n', control_residual_1)
# print('Residuals for check points:\n', control_residual_5check)

# Use 105, 202, 200, 104
# control_2 = control_points_copy.loc[['105', '202', '200', '104']]
# control_result_2 = absolute_orientation(control_2)
# control_residual_2 = residual(control_result_2, control_2)
# print(control_result_2)
# print('Residuals for 4 GCPs data:\n', control_residual_2)

# Use 105, 202, 200, 102
# control_3 = control_points_copy.loc[['105', '202', '200', '102']]
# control_result_3 = absolute_orientation(control_3)
# control_residual_3 = residual(control_result_3, control_3)
# print(control_result_3)
# print('Residuals for 4 GCPs data:\n', control_residual_3)

# Use all of GCPs
control_result_4 = absolute_orientation(control_points_copy)
control_check = pd.concat([control_points_copy, check_points_copy], axis = 0)
control_residual_4 = residual(control_result_4, control_check)
control_residual_5 = residual(control_result_4, check_points_copy)
print(control_result_4)
print('Residuals for 5 GCPs and check points:\n', control_residual_4)
print('Residuals for check points:\n', control_residual_5)

# Convert RO to AO
def convert_to_AO(points, params):
    points = points.iloc[:, -3:]
    points_arr = np.array(points)

    x_m = points_arr[:,0]
    y_m = points_arr[:,1]
    z_m = points_arr[:,2]

    omega, phi, kappa, scale, tx, ty, tz = params.iloc[0]
    M = compute_rotation_matrix(omega, phi, kappa)

    x_o = []
    y_o = []
    z_o = []

    for i in range(len(points)):
        x_o.append(scale * (M[0, 0] * x_m[i] + M[0, 1] * y_m[i] + M[0, 2] * z_m[i]) + tx)
        y_o.append(scale * (M[1, 0] * x_m[i] + M[1, 1] * y_m[i] + M[1, 2] * z_m[i]) + ty)
        z_o.append(scale * (M[2, 0] * x_m[i] + M[2, 1] * y_m[i] + M[2, 2] * z_m[i]) + tz)

    x_o = np.array(x_o)  # Convert to numpy arrays
    y_o = np.array(y_o)
    z_o = np.array(z_o)

    df_o = pd.DataFrame({
        'X (m)': x_o,
        'Y (m)': y_o,
        'Z (m)': z_o
    }, index=points.index)

    df_full = pd.concat([points, df_o], axis=1)

    return df_full

# Convert GCP, check, tie and PC
PC_l = [0,0,0]
by, bz, _, _, _ = tie_params.iloc[0]
PC_r = [bx, by, bz]
PC = pd.DataFrame({
    'x':[PC_l[0],PC_r[0]],
    'y':[PC_l[1],PC_r[1]],
    'z':[PC_l[2],PC_r[2]]
}, index=['PC_l', 'PC_r'])

control_AO = convert_to_AO(control_points, control_result_4)
print('GCPs in absolute orientation:\n', control_AO)

check_AO = convert_to_AO(check_points, control_result_4)
print('check points in absolute orientation:\n', check_AO)

tie_AO = convert_to_AO(tie_points, control_result_4)
print('tie points in absolute orientation:\n', tie_AO)

PC_AO = convert_to_AO(PC, control_result_4)
print('perspective centers in absolute orientation:\n', PC_AO)

# Convert Test data
PC_l_test = [0,0,0]
by_test, bz_test, _, _, _ = test_params.iloc[0]
PC_r_test = [bx_test, by_test, bz_test]
PC_test = pd.DataFrame({
    'x':[PC_l_test[0],PC_r_test[0]],
    'y':[PC_l_test[1],PC_r_test[1]],
    'z':[PC_l_test[2],PC_r_test[2]]
}, index=['PC_l', 'PC_r'])

test_AO = convert_to_AO(test_points, test_result)
print('test points in absolute orientation:\n', test_AO)
PC_test_AO = convert_to_AO(PC_test, test_result)
print('test PC in absolute orientation:\n', PC_test_AO)


# Transform the rotation matrix and extract rotation angles
def O_to_I_matrix(params_IM, params_MO):
    # Make sure each line and column shows
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Calculate Angles and Matrices from RO
    _, _, omega_IM, phi_IM, kappa_IM = params_IM.iloc[0]
    M_IM_L = np.eye(3)
    M_IM_R = compute_rotation_matrix(omega_IM, phi_IM, kappa_IM)
    omega_IM_L, phi_IM_L, kappa_IM_L = [0,0,0]
    M_IM = {
        'Left': {
            'omega_IM_L': omega_IM_L,
           'phi_IM_L': phi_IM_L,
           'kappa_IM_L': kappa_IM_L,
            'M_IM_L': [M_IM_L]  
        },
        'Right': {
            'omega_IM': omega_IM,
            'phi_IM': phi_IM,
            'kappa_IM': kappa_IM,
            'M_IM_R': [M_IM_R]  
        }
    }
    df_IM = pd.DataFrame(M_IM)
    print('Angles and Matrices from RO:\n', df_IM)
    
    # Calculate Matrix from AO and get the transpose
    omega_MO, phi_MO, kappa_MO, _, _, _, _ = params_MO.iloc[0]
    M_MO = compute_rotation_matrix(omega_MO, phi_MO, kappa_MO)
    M_MO_T = M_MO.T
    df_MO = pd.DataFrame({
        'M_MO': [f'{M_MO[0, 0]} {M_MO[0, 1]} {M_MO[0, 2]}',
                 f'{M_MO[1, 0]} {M_MO[1, 1]} {M_MO[1, 2]}',
                 f'{M_MO[2, 0]} {M_MO[2, 1]} {M_MO[2, 2]}'],
        'M_MO_T': [f'{M_MO_T[0, 0]} {M_MO_T[0, 1]} {M_MO_T[0, 2]}',
                   f'{M_MO_T[1, 0]} {M_MO_T[1, 1]} {M_MO_T[1, 2]}',
                   f'{M_MO_T[2, 0]} {M_MO_T[2, 1]} {M_MO_T[2, 2]}']
    })
    print('Matrix from AO:\n', df_MO)
    
    # Calculate matrix products
    M_OI_L = M_IM_L @ M_MO_T
    omega_L = np.rad2deg(np.atan(-(M_OI_L[2,1]/M_OI_L[2,2])))
    phi_L = np.rad2deg(np.asin(M_OI_L[2,0]))
    kappa_L = np.rad2deg(np.atan(-(M_OI_L[1,0]/M_OI_L[0,0])))
    
    M_OI_R = M_IM_R @ M_MO_T
    omega_R = np.rad2deg(np.atan(-(M_OI_R[2,1]/M_OI_R[2,2])))
    phi_R = np.rad2deg(np.asin(M_OI_R[2,0]))
    kappa_R = np.rad2deg(np.atan(-(M_OI_R[1,0]/M_OI_R[0,0])))
    
    df_OI = pd.DataFrame({
        'M_OI_L': [f'{M_OI_L[0, 0]} {M_OI_L[0, 1]} {M_OI_L[0, 2]}',
                   f'{M_OI_L[1, 0]} {M_OI_L[1, 1]} {M_OI_L[1, 2]}',
                   f'{M_OI_L[2, 0]} {M_OI_L[2, 1]} {M_OI_L[2, 2]}'],
        'M_OI_R': [f'{M_OI_R[0, 0]} {M_OI_R[0, 1]} {M_OI_R[0, 2]}',
                   f'{M_OI_R[1, 0]} {M_OI_R[1, 1]} {M_OI_R[1, 2]}',
                   f'{M_OI_R[2, 0]} {M_OI_R[2, 1]} {M_OI_R[2, 2]}']
    })
    print('Resultant matrix products:\n', df_OI)
    
    M_OI = {
        'Left': {
            'omega_OI_L (deg)': omega_L,
           'phi_OI_L (deg)': phi_L,
           'kappa_OI_L (deg)': kappa_L,
        },
        'Right': {
            'omega_OI_R (deg)': omega_R,
            'phi_OI_R (deg)': phi_R,
            'kappa_OI_R (deg)': kappa_R,
        }
    }
    
    df_Angle = pd.DataFrame(M_OI)
    print('Extracted Angles:\n', df_Angle)
    
# For test data
test_matrix = O_to_I_matrix(test_params, test_result)
print(test_matrix)

# For lab data
lab_matrix = O_to_I_matrix(tie_params, control_result_4)
print(lab_matrix)


















# Lab5: To perform single photo resection (SPR) and space intersection as part of the overarching aim of labs 2-5 of photogrammetric restitution
print('\n' * 5)
print('Lab5: To perform single photo resection (SPR) and space intersection as part of the overarching aim of labs 2-5 of photogrammetric restitution')
# Part a
# Determine EO parameters per image

# Recall three dataframe
# Each of them has 11 columns, xl (mm), yl (mm), xr (mm), yr (mm), pY (mm), X (mm), Y (mm), Z (mm), X (m), Y (m), Z (m)
test_points_copy = test_points_copy
check_points_copy = check_points_copy 
control_points_copy = control_points_copy 

