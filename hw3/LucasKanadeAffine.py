import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.copy(M).flatten()

    delta_p = np.zeros((2,3))
    rect_spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    rect_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)


    for i in range(0, int(num_iters)):    
        if i==0 or np.sum(delta_p ** 2) >= threshold:
        # if i==0 or (delta_p[0]**2+delta_p[1]**2+delta_p[2]**2+delta_p[3]**2+delta_p[4]**2+delta_p[5]**2)**0.5 >= threshold:
            # print (delta_p.shape)
        # print(i)
        # if True:
            x_rect_it1 = np.arange(0, It1.shape[0])
            y_rect_it1 = np.arange(0, It1.shape[1])
            X_rect_it1, Y_rect_it1 = np.meshgrid(x_rect_it1, y_rect_it1)
            X_new = p[0] * X_rect_it1 + p[2] * Y_rect_it1 + p[4]
            Y_new = p[1] * X_rect_it1 + p[3]* Y_rect_it1 + p[5]
            interpolated_It1 = rect_spline_It1.ev(Y_new, X_new)

            x_rect_it = np.arange(0, It1.shape[0])
            y_rect_it = np.arange(0, It1.shape[1])
            X_rect_it, Y_rect_it = np.meshgrid(x_rect_it, y_rect_it)
            interpolated_It = rect_spline_It.ev(Y_rect_it, X_rect_it)

            gradient_x = rect_spline_It1.ev( Y_new, X_new, dx=0, dy=1).flatten()
            gradient_y = rect_spline_It1.ev( Y_new, X_new, dx=1, dy=0).flatten()

            N = gradient_x.shape[0]

            A = np.zeros((N, 6))
            A[:, 0] = X_new.flatten() * gradient_x
            A[:, 1] = X_new.flatten() * gradient_y
            A[:, 2] = Y_new.flatten() * gradient_x
            A[:, 3] = Y_new.flatten() * gradient_y
            A[:, 4] = gradient_x
            A[:, 5] = gradient_y

            # print (A.shape)
            # A = np.transpose(A)
            b = (interpolated_It - interpolated_It1).flatten()
            # print("b", b.shape)
            # b = np.transpose(b)

            # print( np.transpose(A) @ A ) @ ( np.transpose(A) @ b )
            # print("hello")
            delta_p = np.linalg.pinv(A) @ b
            p = p + delta_p.T
            # p = [[1 + p[0]+delta_p[0], p[2]+delta_p[2],  p[4]+delta_p[4]], [ p[1]+delta_p[1],  1+ p[3]+delta_p[3],  p[5]+delta_p[5]]]

        else:
            break
    M=[[1 + p[0]+delta_p[0], p[2]+delta_p[2],  p[4]+delta_p[4]], [ p[1]+delta_p[1],  1+ p[3]+delta_p[3],  p[5]+delta_p[5]], [-0,-0,-1]]
    return M
