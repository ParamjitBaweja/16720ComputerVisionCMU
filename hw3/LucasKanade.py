import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # print(It.shape)
    # print(It1.shape)
	

    delta_p = np.zeros((1, 2))
    p = p0
    rect_spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    rect_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    for i in range(0, int(num_iters)):    
        if i==0 or (delta_p[0]**2+delta_p[1]**2)**0.5 >= threshold:
            x_rect_it1 = np.arange(rect[0] + p[0], rect[2] + 0.001 + p[0])
            y_rect_it1 = np.arange(rect[1] + p[1], rect[3] + 0.001 + p[1])
            X_rect_it1, Y_rect_it1 = np.meshgrid(x_rect_it1, y_rect_it1)
            interpolated_It1 = rect_spline_It1.ev(Y_rect_it1, X_rect_it1)

            x_rect_it = np.arange(rect[0], rect[2] + 0.001)
            y_rect_it = np.arange(rect[1], rect[3] + 0.001)
            X_rect_it, Y_rect_it = np.meshgrid(x_rect_it, y_rect_it)
            interpolated_It = rect_spline_It.ev(Y_rect_it, X_rect_it)

            gradient_y = rect_spline_It1.ev(Y_rect_it1, X_rect_it1, dx=0, dy=1).flatten()
            gradient_x = rect_spline_It1.ev(Y_rect_it1, X_rect_it1, dx=1, dy=0).flatten()

            N = gradient_x.shape[0]

            A = np.zeros((N, 2))
            A[:, 0] = gradient_y
            A[:, 1] = gradient_x

            b = (interpolated_It - interpolated_It1).flatten()

            # delta_p = np.linalg.inv( np.transpose(A) @ A ) @  np.transpose(A) @ b
            delta_p = np.linalg.pinv(A)@b 

            p = [p[0]+delta_p[0], p[1]+delta_p[1]]

        else:
            break
    return p
