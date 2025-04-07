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
], columns=['x(pixel)', 'y(pixel)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

fiducial_digital_28 = pd.DataFrame(data=[
    [1345.25,-19285],
    [19173.75,-1482.75],
    [1357,-1471],
    [19162.5,-19296.25],
    [848,-10378],
    [19672.25,-10390],
    [10267.25,-972],
    [10253.25,-19794.5]
], columns=['x(pixel)', 'y(pixel)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

fiducial_true = pd.DataFrame(data=[
    [-105.997,-105.995],
    [106.004,106.008],
    [-106.000,106.009],
    [106.012,-105.995],
    [-112.000,0.007],
    [112.006,0.007],
    [0.005,112.007],
    [0.002,-111.998]
], columns=['x(pixel)', 'y(pixel)'], index=['1', '2', '3', '4', '5', '6', '7', '8'])

# tie points
tie_digital_27 = pd.DataFrame(data=[
    [9112.75,-10878.25],
    [19254.5,-9756],
    [8524.75,-16100.25],
    [19133.75,-16965],
    [9192.25,-3894.5],
    [18587.5,-4037.75]
], columns=['x(pixel)', 'y(pixel)'], index=['1', '2', '3', '4', '5', '6'])

tie_digital_28 = pd.DataFrame(data=[
    [1346.75,-10810],
    [11233.5,-9382.25],
    [912.5,-16032.25],
    [11398.5,-16602.25],
    [1197,-3730.5],
    [10654,-3575]
], columns=['x(pixel)', 'y(pixel)'], index=['1', '2', '3', '4', '5', '6'])

# Control points
control_digital_27 = pd.DataFrame(data=[
    [10059,-10881.75],
    [11844,-17251.25],
    [17841.25,-18026.25],
    [9612.25,-14502.25],
    [11779.75,-1174]
], columns=['x(pixel)', 'y(pixel)'], index=['102', '104', '105', '202', '200'])

control_digital_28 = pd.DataFrame(data=[
    [2275.25,-10786],
    [4159.5,-17082.75],
    [10136.5,-17687.75],
    [1949.5,-14416.75],
    [3724.75,-853.25]
], columns=['x(pixel)', 'y(pixel)'], index=['102', '104', '105', '202', '200'])

# Check points
check_digital_27 = pd.DataFrame(data=[
    [14685.75,-18204.75],
    [14006.25,-9748.5],
    [9460.25,-2291.5]
], columns=['x(pixel)', 'y(pixel)'], index=['203', '201', '100'])

check_digital_28 = pd.DataFrame(data=[
    [6984,-17948.75],
    [6158.75,-9527.5],
    [1411,-2079.75]
], columns=['x(pixel)', 'y(pixel)'], index=['203', '201', '100'])

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
    return params, residuals_x, residuals_y, RMSE_x, RMSE_y

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
params1, residual_x_27, residual_y_27, rmse_x_27, rmse_y_27 = result_27
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
params2, residual_x_28, residual_y_28, rmse_x_28, rmse_y_28 = result_28
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
# test_points_copy, check_points_copy, control_points_copy

# Define a function to get initial values
def initial_value_EOPs(points, c):
    """_summary_

    Args:
        points (_dataframe_): _columns from left to right are xy(mm) of image points and xyz(m) of GCPs_
        c (_number_): _focal length of camera in mm_
    """
    # Get data from points
    x = points.iloc[:, 0]
    y = points.iloc[:, 1]
    X = points.iloc[:, 2]
    Y = points.iloc[:, 3]
    Z = points.iloc[:, 4]
    Z_arr = np.array(Z)
    print(points)
    # set up design matrix A and observation vector l
    A = np.zeros((2 * len(x), 4))
    l = np.zeros(2 * len(x))
    
    # fill A and l
    for i in range(len(x)):
        A[2*i, 0] = x.iloc[i] / 1000 # a
        A[2*i, 1] = -y.iloc[i] / 1000  # b
        A[2*i, 2] = 1  # dx
        A[2*i, 3] = 0  # dy
        
        A[2*i+1, 0] = y.iloc[i] / 1000  # a
        A[2*i+1, 1] = x.iloc[i] / 1000  # b
        A[2*i+1, 2] = 0        # dx
        A[2*i+1, 3] = 1        # dy
        
        l[2*i] = X.iloc[i]
        l[2*i+1] = Y.iloc[i]
        
    A_T_A = A.T @ A
    A_T_l = A.T @ l
    A_T_A_inv = np.linalg.inv(A_T_A)
        
    params = A_T_A_inv @ A_T_l
    a, b, dx, dy = params
    k = np.arctan2(b,a)
    k = np.rad2deg(k)
    x0 = dx
    y0 = dy
    scale = np.sqrt(a**2 + b**2)
    z0 = c / 1000 * scale + np.average(Z_arr)
    
    initial_values = [x0, y0, z0, 0, 0, k]
    params_names = ['x', 'y', 'z', 'omega', 'phi', 'kappa']
    result_df = pd.DataFrame(initial_values, index = params_names, columns=['Value'])
    
    return result_df, scale

# Get Initial values for left image
left_Resction = control_points_copy.loc[['104', '105', '202', '200'], ['xl (mm)', 'yl (mm)', 'X (m)', 'Y (m)', 'Z (m)']]
left_initial, scale_left = initial_value_EOPs(left_Resction, c)
print("Initial values for left image resction is:\n", left_initial)

# Get Initial values for right image
right_Resction = control_points_copy.loc[['104', '105', '202', '200'], ['xr (mm)', 'yr (mm)', 'X (m)', 'Y (m)', 'Z (m)']]
right_initial, scale_right = initial_value_EOPs(right_Resction, c)
print("Initial values for right image resction is:\n", right_initial)

# Get Initial values for test data
test_Resction = test_points_copy.loc[['30', '40', '50', '112'], ['xl (mm)', 'yl (mm)', 'X (m)', 'Y (m)', 'Z (m)']]
test_initial, scale_test = initial_value_EOPs(test_Resction, c_test)
print("Initial values for test resction is:\n", test_initial)

# Define a function to get tolerance
def tolerance(sigma, scale, side_length_x, side_length_y, c):
    """_summary_

    Args:
        sigma (_float_): _RMSE x or RMSE y or half of pixel in mm_
        scale (_float_): _convert image coords in m to object coords in m_
        side_length_x (_type_): _the length of image in mm_
        side_length_y (_type_): _the length of image in mm_
    """
    Tol_coords = scale * sigma / 1000 / 10
    Tol_tilt = sigma / 10 / c
    Tol_tilt = np.rad2deg(Tol_tilt)
    r_max = np.sqrt(side_length_x ** 2 + side_length_y ** 2) / 2
    Tol_k = sigma/(10*r_max)
    Tol_k = np.rad2deg(Tol_k)
    data = [Tol_coords, Tol_tilt, Tol_k]
    df = pd.DataFrame(data=data, index=['Tol_coords(m)', 'Tol_tilt(deg)', 'Tol_k(deg)'], columns=['Value'])
    
    return df

# Get tolerances for left image
side_length_x = 212
side_length_y = 212
tolerance_left = tolerance(rmse_x_27, scale_left, side_length_x, side_length_y, c)
print("Tolerance for left image is:\n", tolerance_left)

# Get tolerances for right image
tolerance_right = tolerance(rmse_x_28, scale_right, side_length_x, side_length_y, c)
print("Tolerance for right image is:\n", tolerance_right)

# Get tolerances for test data
sigma_test = 0.015 # unit is mm
side_length_x_test = 229 # unit is mm
side_length_y_test = 229
tolerance_test = tolerance(sigma_test, scale_test, side_length_x_test, side_length_y_test, c_test)
print("Tolerance for test data is:\n", tolerance_test)

# Define a function to calculate EOPs
def resection(points, sigma, c, initials, tolerance):
    """_summary_

    Args:
        points (_dataframe_): _columns from left to right are xy(mm) of image points and xyz(m) of GCPs_
        sigma (_number or list_): _if it's number, apply stochastic model; if it's list, build a new P matrix, unit is mm_
        c (_number_): _focal length of camera in mm_
    """
    
    # Print the title
    frame = inspect.currentframe()
    caller_locals = frame.f_back.f_locals  # Get the local variables from the caller's frame
    for var_name, var_value in caller_locals.items():
        if var_value is points:  # Match the value passed with the name
            print(f"Results for {var_name}:\n")
            break
    
    # Observations
    x = np.array(points.iloc[:, 0])
    y = np.array(points.iloc[:, 1])
    X = np.array(points.iloc[:, 2])
    Y = np.array(points.iloc[:, 3])
    Z = np.array(points.iloc[:, 4])
    
    # Initial values
    x_c, y_c, z_c, omega0, phi0, kappa0 = initials.iloc[:,0]
    tol_coords, tol_tilt, tol_k = tolerance.iloc[:,0]
    
    # Set weight matrix P
    P = np.eye(2 * len(x))
    if isinstance(sigma, list):
        for i in range(len(x)):
            P[2*i, 2*i] = 1/(sigma[0]**2)
            P[2*i+1, 2*i+1] = 1/(sigma[1]**2)
    else:
        P = 1/(sigma**2) * P
    
    iter = 100  
    # Set iteration loop
    for i in range(iter):
        # Set up design matrix and misclosure matrix
        A = np.zeros((2 * len(x), 6))
        w = np.zeros(2 * len(x))
        
        # Rotation matrix
        R = compute_rotation_matrix(omega0, phi0, kappa0)
        omega, phi, kappa = np.radians([omega0, phi0, kappa0])  # Convert to radians
        
        # Fill A and w
        for j in range(len(x)):
            U = R[0,0] * (X[j] - x_c) + R[0,1] * (Y[j] - y_c) + R[0,2] * (Z[j] - z_c)
            V = R[1,0] * (X[j] - x_c) + R[1,1] * (Y[j] - y_c) + R[1,2] * (Z[j] - z_c)
            W = R[2,0] * (X[j] - x_c) + R[2,1] * (Y[j] - y_c) + R[2,2] * (Z[j] - z_c)
            
            w[2*j] =  - c * U / W - x[j]
            w[2*j+1] =  - c * V / W - y[j]
            
            A[2*j,0] = - c * (U * R[2,0] - W * R[0,0]) / W**2 # xc
            A[2*j,1] = - c * (U * R[2,1] - W * R[0,1]) / W**2 # yc
            A[2*j,2] = - c * (U * R[2,2] - W * R[0,2]) / W**2 # zc
            A[2*j,3] = - c * ((Y[j] - y_c) * (U * R[2,2] - W * R[0,2]) - (Z[j] - z_c) * (U * R[2,1] - W * R[0,1])) / W**2 # omega
            A[2*j,4] = - c * ((X[j] - x_c) * (- W * np.sin(phi) * np.cos(kappa) - U * np.cos(phi)) + 
                                (Y[j] - y_c) * (W * np.sin(omega) * np.cos(phi) * np.cos(kappa) - U * np.sin(omega) * np.sin(phi)) + 
                                (Z[j] - z_c) * (- W * np.cos(omega) * np.cos(phi) * np.cos(kappa) + U * np.cos(omega) * np.sin(phi))) / W**2 # phi
            A[2*j,5] = - c * V / W # kappa
            
            
            A[2*j+1,0] = - c * (V * R[2,0] - W * R[1,0]) / W**2
            A[2*j+1,1] = - c * (V * R[2,1] - W * R[1,1]) / W**2
            A[2*j+1,2] = - c * (V * R[2,2] - W * R[1,2]) / W**2
            A[2*j+1,3] = - c * ((Y[j] - y_c) * (V * R[2,2] - W * R[1,2]) - (Z[j] - z_c) * (V * R[2,1] - W * R[1,1])) / W**2 # omega
            A[2*j+1,4] = - c * ((X[j] - x_c) * (W * np.sin(phi) * np.sin(kappa) - V * np.cos(phi)) + 
                                (Y[j] - y_c) * (- W * np.sin(omega) * np.cos(phi) * np.sin(kappa) - V * np.sin(omega) * np.sin(phi)) + 
                                (Z[j] - z_c) * (W * np.cos(omega) * np.cos(phi) * np.sin(kappa) + V * np.cos(omega) * np.sin(phi))) / W**2 #phi
            A[2*j+1,5] = c * U / W # kappa
        
        # print(P)  
        H = A.T @ P @ A
        g = A.T @ P @ w
        C = np.linalg.inv(H)
        delta = -C @ g
        
        
        print(f"the {i+1}th iteration:\n")
        # Updata params
        x_c_new = x_c + delta[0]
        y_c_new = y_c + delta[1]
        z_c_new = z_c + delta[2]
        omega_new = np.rad2deg(omega + delta[3])
        phi_new = np.rad2deg(phi + delta[4])
        kappa_new = np.rad2deg(kappa + delta[5])
        params_df = pd.DataFrame(
            {'Parameters':[x_c_new, y_c_new, z_c_new, omega_new, phi_new, kappa_new],
            'delta':delta
            }, index=['xc (m)', 'yc (m)', 'zc (m)', 'omega (deg)', 'phi (deg)', 'kappa (deg)']
        )
        print('Parameters and delta:\n', params_df)
        
        A_df = pd.DataFrame(data=A, columns=['xc', 'yc', 'zc', 'omega', 'phi', 'kappa'])
        w_df = pd.DataFrame(data=w, columns=['w vector (mm)'])
        print("A matrix:\n", A_df)
        print('w matrix:\n', w_df)
        
        # Check tolerance
        error_coords = np.sqrt((x_c_new - x_c)**2+(y_c_new - y_c)**2+(z_c_new - z_c)**2)
        error_tilt = np.sqrt((omega_new - omega0)**2+(phi_new - phi0)**2)
        if error_coords < tol_coords and error_tilt < tol_tilt and abs(kappa_new - kappa0) < tol_k:
            break
        
        x_c, y_c, z_c, omega0, phi0, kappa0 = x_c_new, y_c_new, z_c_new, omega_new, phi_new, kappa_new
    
    # Compute correlation matrix
    diag_std = np.sqrt(np.diag(C))
    correlation_matrix = C / np.outer(diag_std, diag_std)
    C_df = pd.DataFrame(correlation_matrix, columns=['xc', 'yc', 'zc', 'omega', 'phi', 'kappa'], index=['xc', 'yc', 'zc', 'omega', 'phi', 'kappa'])
    print('Correlation matrix:\n', C_df)
    
    # Compute STD
    std = np.sqrt(np.diag(C))
    std_df = pd.DataFrame(std, index=['xc (m)', 'yc (m)', 'zc (m)', 'omega (deg)', 'phi (deg)', 'kappa (deg)'], columns=['STD'])
    print("STD is:\n", std_df)

    # Compute redundancy number
    redundancy = 2 * len(x) - 6
    residual = w.reshape(4,2)
    RMSE_x = np.sqrt((residual[:,0] ** 2).mean())
    RMSE_y = np.sqrt((residual[:,1] ** 2).mean())
    if redundancy != 0:
        varFactor = 1 / redundancy * np.sum(w**2)
    else:
        varFactor = np.nan
    other_df = pd.DataFrame(data=[RMSE_x, RMSE_y, redundancy, varFactor], index=['RMSE_x (mm)', 'RMSE_y (mm)', 'Redundancy', 'Variance factor'])
    print('Other quantities:\n', other_df)
        
    params_df = pd.DataFrame(
                {'Parameters':[x_c, y_c, z_c, omega0, phi0, kappa0]
                }, index=['xc (m)', 'yc (m)', 'zc (m)', 'omega (deg)', 'phi (deg)', 'kappa (deg)']
            )
            
    return params_df

# Compute EOPs for left image
sigma_left = [rmse_x_27, rmse_y_27]
left_EOPs = resection(left_Resction, sigma_left, c, left_initial, tolerance_left)
print(left_EOPs)

# Compute EOPs for right image
sigma_right = [rmse_x_28, rmse_y_28]
right_EOPs = resection(right_Resction, sigma_right, c, right_initial, tolerance_right)
print(right_EOPs)

# Use test data to check
test_EOPs = resection(test_Resction, sigma_test, c_test, test_initial, tolerance_test)
print(test_EOPs)



# Part B

# Conduct measurements
# Get origin data from excel
file_path = 'chosen points.xlsx'
tennis_court_27 = pd.read_excel(file_path, sheet_name=0, usecols="F,G", header=None, skiprows=3, nrows=16)
tennis_court_27.columns = ['xl (pixel)', 'yl (pixel)']
tennis_court_27.index = range(1, 17)

tennis_court_28 = pd.read_excel(file_path, sheet_name=0, usecols="M,N", header=None, skiprows=3, nrows=16)
tennis_court_28.columns = ['xr (pixel)', 'yr (pixel)']
tennis_court_28.index = range(1, 17)

# Refine data
tennis_true_27 = to_fiducial_system(params1, tennis_court_27)
tennis_true_28 = to_fiducial_system(params2, tennis_court_28)

tennis_27_corrected = systematic_error(tennis_true_27, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
tennis_28_corrected = systematic_error(tennis_true_28, principal_offset, radial_distortion, decentering_distortion, atmospheric_refraction)
tennis_corrected = pd.concat([tennis_27_corrected, tennis_28_corrected], axis=1)
print('correct tennis points coords:\n', tennis_corrected)

# Intersect
# Define a function to get initial values for IOPs
def initial_value_IOPs(points, left_EOPs, right_EOPs, c):
    
    # Get coords from points
    xl = np.array(points.iloc[:,0])
    yl = np.array(points.iloc[:,1])
    xr = np.array(points.iloc[:,2])
    yr = np.array(points.iloc[:,3])
    
    # Get EOPs
    xc_l, yc_l, zc_l, omega_l, phi_l, kappa_l = left_EOPs.iloc[:,0]
    xc_r, yc_r, zc_r, omega_r, phi_r, kappa_r = right_EOPs.iloc[:,0]
    
    # Compute rotation matrix
    R_l = compute_rotation_matrix(omega_l, phi_l, kappa_l)
    R_r = compute_rotation_matrix(omega_r, phi_r, kappa_r)
    
    # Create a list to store the parameters for each point
    params_list = []
    
    # Compute Initial values for each point
    for i in range(len(xl)):
        # set up design matrix A and observation vector l
        A = np.zeros((4 * len(xl), 3))
        l = np.zeros(4 * len(xl))
        
        # Fill A and l
        A[4*i, 0] = xl[i] * R_l[2,0] + c * R_l[0,0]
        A[4*i, 1] = xl[i] * R_l[2,1] + c * R_l[0,1]
        A[4*i, 2] = xl[i] * R_l[2,2] + c * R_l[0,2]
        l[4*i] = A[4*i, 0] * xc_l + A[4*i, 1] * yc_l + A[4*i, 2] * zc_l
        
        A[4*i+1, 0] = yl[i] * R_l[2,0] + c * R_l[1,0]
        A[4*i+1, 1] = yl[i] * R_l[2,1] + c * R_l[1,1]
        A[4*i+1, 2] = yl[i] * R_l[2,2] + c * R_l[1,2]
        l[4*i+1] = A[4*i+1, 0] * xc_l + A[4*i+1, 1] * yc_l + A[4*i+1, 2] * zc_l
        
        A[4*i+2, 0] = xr[i] * R_r[2,0] + c * R_r[0,0]
        A[4*i+2, 1] = xr[i] * R_r[2,1] + c * R_r[0,1]
        A[4*i+2, 2] = xr[i] * R_r[2,2] + c * R_r[0,2]
        l[4*i+2] = A[4*i+2, 0] * xc_r + A[4*i+2, 1] * yc_r + A[4*i+2, 2] * zc_r
        
        A[4*i+3, 0] = yr[i] * R_r[2,0] + c * R_r[1,0]
        A[4*i+3, 1] = yr[i] * R_r[2,1] + c * R_r[1,1]
        A[4*i+3, 2] = yr[i] * R_r[2,2] + c * R_r[1,2]
        l[4*i+3] = A[4*i+3, 0] * xc_r + A[4*i+3, 1] * yc_r + A[4*i+3, 2] * zc_r
        
        A_T_A = A.T @ A
        A_T_l = A.T @ l
        A_T_A_inv = np.linalg.inv(A_T_A)
            
        params = A_T_A_inv @ A_T_l
        
        # Append the result as a series with the point index as the column name
        params_list.append(params)
    
    # Convert the list of parameters into a DataFrame
    # Each column corresponds to a point (based on the index of points)
    params_df = pd.DataFrame(data=np.column_stack(params_list), 
                             index=['X0', 'Y0', 'Z0'], 
                             columns=points.index)
    
    return params_df

# Use test data to check
test_Intersection = test_points_copy.loc[['72', '127'], ['xl (mm)', 'yl (mm)','xr (mm)', 'yr (mm)', 'X (m)', 'Y (m)', 'Z (m)']]
testEOPs_left = pd.DataFrame(data=[6349.488,3965.252, 1458.095, 0.9885, 0.4071, -18.9049], index=['xc (m)', 'yc (m)', 'zc (m)', 'omega (deg)', 'phi (deg)', 'kappa (deg)'])
testEOPs_right = pd.DataFrame(data=[7021.897,3775.68, 1466.702, 1.8734, 1.6751, -15.7481], index=['xc (m)', 'yc (m)', 'zc (m)', 'omega (deg)', 'phi (deg)', 'kappa (deg)'])
test_initial_IOPs= initial_value_IOPs(test_Intersection, testEOPs_left, testEOPs_right, c_test)
print("Initial values for test intersection is:\n", test_initial_IOPs)

# Get initial IOPs for tennis court data
tennis_initial_IOPs = initial_value_IOPs(tennis_corrected, left_EOPs, right_EOPs, c)
print("Initial values for lab intersection is:\n", tennis_initial_IOPs)

# Define a function to calculate IOPs
def intersection(points, sigma, c, left_EOPs, right_EOPs, initials, tolerance):
    

    # Print the title
    frame = inspect.currentframe()
    caller_locals = frame.f_back.f_locals  # Get the local variables from the caller's frame
    for var_name, var_value in caller_locals.items():
        if var_value is points:  # Match the value passed with the name
            print(f"Results for {var_name}:\n")
            break
    
    # Observations
    xl = np.array(points.iloc[:, 0])
    yl = np.array(points.iloc[:, 1])
    xr = np.array(points.iloc[:, 2])
    yr = np.array(points.iloc[:, 3])
    
    # Initial values
    X0, Y0, Z0 = initials.iloc[:,0]
    tol_coords, tol_tilt, tol_k = tolerance.iloc[:,0]
    
    # Get EOPs
    xc_l, yc_l, zc_l, omega_l, phi_l, kappa_l = left_EOPs.iloc[:,0]
    xc_r, yc_r, zc_r, omega_r, phi_r, kappa_r = right_EOPs.iloc[:,0]
    
    # Compute rotation matrix
    R_l = compute_rotation_matrix(omega_l, phi_l, kappa_l)
    R_r = compute_rotation_matrix(omega_r, phi_r, kappa_r)
    
    # Set weight matrix P
    P = np.eye(4)
    if isinstance(sigma, list):
        P[0,0] = 1/(sigma[0]**2) # rmse_x_left
        P[1,1] = 1/(sigma[1]**2) # rmse_y_left
        P[2,2] = 1/(sigma[2]**2) # rmse_x_right
        P[3,3] = 1/(sigma[3]**2) # rmse_y_right
            
    else:
        P = 1/(sigma**2) * P
        
    # Create empty DataFrames to store results
    redundancy_df = pd.DataFrame(columns=[i+1 for i in range(len(xl))], index=['x_left', 'y_left', 'x_right', 'y_right'])
    IOPs_df = pd.DataFrame(columns=['X (m)', 'Y (m)', 'Z (m)'])
        
    # Set points loop
    for i in range(len(xl)):
        print(f'the {i+1}th point:\n')
        iter = 4
        for j in range(iter):
            # Set up design matrix and misclosure matrix
            A = np.zeros((4, 3))
            w = np.zeros(4)
            
            # Fill A and w
            # Compute UVM for each image
            Ul = R_l[0,0] * (X0 - xc_l) + R_l[0,1] * (Y0 - yc_l) + R_l[0,2] * (Z0 - zc_l)
            Vl = R_l[1,0] * (X0 - xc_l) + R_l[1,1] * (Y0 - yc_l) + R_l[1,2] * (Z0 - zc_l)
            Wl = R_l[2,0] * (X0 - xc_l) + R_l[2,1] * (Y0 - yc_l) + R_l[2,2] * (Z0 - zc_l)
            
            Ur = R_r[0,0] * (X0 - xc_r) + R_r[0,1] * (Y0 - yc_r) + R_r[0,2] * (Z0 - zc_r)
            Vr = R_r[1,0] * (X0 - xc_r) + R_r[1,1] * (Y0 - yc_r) + R_r[1,2] * (Z0 - zc_r)
            Wr = R_r[2,0] * (X0 - xc_r) + R_r[2,1] * (Y0 - yc_r) + R_r[2,2] * (Z0 - zc_r)
            
            # Compute w
            w[0] =  - c * Ul / Wl - xl[i]
            w[1] =  - c * Vl / Wl - yl[i]
            w[2] =  - c * Ur / Wr - xr[i]
            w[3] =  - c * Vr / Wr - yr[i]
            
            # Compute A
            A[0,0] = c * (Ul * R_l[2,0] - Wl * R_l[0,0]) / Wl**2 # x
            A[0,1] = c * (Ul * R_l[2,1] - Wl * R_l[0,1]) / Wl**2 # y
            A[0,2] = c * (Ul * R_l[2,2] - Wl * R_l[0,2]) / Wl**2 # z

            A[1,0] = c * (Vl * R_l[2,0] - Wl * R_l[1,0]) / Wl**2
            A[1,1] = c * (Vl * R_l[2,1] - Wl * R_l[1,1]) / Wl**2
            A[1,2] = c * (Vl * R_l[2,2] - Wl * R_l[1,2]) / Wl**2
            
            A[2,0] = c * (Ur * R_r[2,0] - Wr * R_r[0,0]) / Wr**2 # x
            A[2,1] = c * (Ur * R_r[2,1] - Wr * R_r[0,1]) / Wr**2 # y
            A[2,2] = c * (Ur * R_r[2,2] - Wr * R_r[0,2]) / Wr**2 # z

            A[3,0] = c * (Vr * R_r[2,0] - Wr * R_r[1,0]) / Wr**2
            A[3,1] = c * (Vr * R_r[2,1] - Wr * R_r[1,1]) / Wr**2
            A[3,2] = c * (Vr * R_r[2,2] - Wr * R_r[1,2]) / Wr**2
            
            H = A.T @ P @ A
            g = A.T @ P @ w
            C = np.linalg.inv(H)
            delta = -C @ g

            # Updata params
            X0_new = X0 + delta[0]
            Y0_new = Y0 + delta[1]
            Z0_new = Z0 + delta[2]

            if len(xl) < 3:
                print(f"the {j+1}th iteration:\n")
                A_df = pd.DataFrame(data=A, columns=['X0', 'Y0', 'Z0'], index=['xl', 'yl', 'xr', 'yr'])
                w_df = pd.DataFrame(data=w, columns=['w vector (mm)'])
                print("A matrix:\n", A_df)
                print('w matrix:\n', w_df)
            
            # Check tolerance
            error_coords = np.sqrt((X0_new - X0)**2+(Y0_new - Y0)**2+(Z0_new - Z0)**2)
            if error_coords < tol_coords:
                break
        
            X0, Y0, Z0 = X0_new, Y0_new, Z0_new
            
        # Compute residual
        residual = w.reshape(2,2)
        RMSE_x = np.sqrt((residual[:,0] ** 2).mean())
        RMSE_y = np.sqrt((residual[:,1] ** 2).mean())
        rmse_row = np.array([RMSE_x, RMSE_y])
        residual_with_rmse = np.vstack([residual, rmse_row])
        residual_df = pd.DataFrame(data=residual_with_rmse, columns=['residual_x (mm)', 'residual_y (mm)'])
        print('Residual vector:\n', residual_df)
        
        # Compute redundancy number
        I = np.eye(4)
        C_x = np.linalg.inv(H)
        R = I - A @ C_x @ A.T @ P
        diagonal = [R[i, i] for i in range(4)]
        diagonal_reshape = np.array(diagonal).reshape(-1,1)
        df_R = pd.DataFrame(diagonal_reshape, index=['x_left', 'y_left', 'x_right', 'y_right'])
        df_R.loc['Total'] = df_R.sum()
        
        # Compute STD
        std = np.sqrt(np.diag(C_x))
        std_df = pd.DataFrame(std, index=['xc (m)', 'yc (m)', 'zc (m)'], columns=['STD'])
        print("STD is:\n", std_df)

        # Store redundancy number for the current point
        redundancy_df[i+1] = diagonal
        print("Redundancy number for each coord:\n", redundancy_df)
            
        # Store the IOPs for the current point
        IOPs_df.loc[i+1] = [X0, Y0, Z0]
        print("IOPs:\n", IOPs_df)  
    return IOPs_df

# Use test data to check
test_IOPs = intersection(test_Intersection, sigma_test, c_test, testEOPs_left, testEOPs_right, test_initial_IOPs, tolerance_test)
print(test_IOPs)

# Compute IOPs for tennis court
sigma = [rmse_x_27, rmse_y_27, rmse_x_28, rmse_y_28]
tennis_IOPS = intersection(tennis_corrected, sigma, c, left_EOPs, right_EOPs, tennis_initial_IOPs, tolerance_right)
print(tennis_IOPS)

# Define a function to compute residuals
def height_residual(IOPs):
    Z = IOPs.iloc[:,2]
    mean_Z = Z.mean()
    residual = Z - mean_Z
    rmse = np.sqrt((residual ** 2).mean())
    
    residual.loc['RMSE'] = [rmse]
    
    return residual

# Compute RMSE for tennis court
tennis_RMSE = height_residual(tennis_IOPS)
print('Residuals and RMSE for lab points:\n', tennis_RMSE)
    