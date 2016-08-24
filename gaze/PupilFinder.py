import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class PupilFinder:

    def __init__(self, alpha = 0.05, max_iterations = 100):
        self.alpha = alpha
        self.max_iterations = max_iterations
        plt.ion()
        plt.show()


    def gradient_ascent(self, eye, c, gx, gy):
        eye_len = np.arange(eye.shape[0])
        X,Y = np.meshgrid(eye_len, eye_len)
        X,Y = [xx-c[0],yy-c[1]]
        norm_dir = np.sqrt(X**2 + Y**2)

        e = (X*gx + Y*gy)

        D1 = (X * e**2 - gx * e * norm_dir**2) / norm_dir**4
        D2 = (Y * e**2 - gy * e * norm_dir**2) / norm_dir**4

        # self.plot_grad(D1, D2)

        d1 = 2/float(D1.size) * np.nansum(D1)
        d2 = 2/float(D2.size) * np.nansum(D2)

        last_difference = sqrt((self.alpha*d1)**2 + (self.alpha*d2)**2)
        return (last_difference, [c[0] + self.alpha*d1,c[1] + self.alpha*d2])


    def find_pupil(self, eye):
        Gx,Gy = np.gradient(eye.astype('float32'))
        G_magn = np.sqrt(Gx**2 + Gy**2)
        c_pos = np.argmax(G_magn)
        c = [Gx.flat[c_pos],Gy.flat[c_pos]]
        last_difference = float('inf')
        num_iterations = 0

        while (num_iterations < self.max_iterations) and (last_difference > 0.03):
            last_difference,c = self.gradient_ascent(eye, c, Gx, Gy)
            num_iterations += 1

        return c

    def plot_for_testing(self, eye):
        eye = cv2.GaussianBlur(eye,(5,5),0.01*eye.shape[1])

        xx,yy = np.meshgrid(eye.shape[0],eye.shape[0])



    def plot_grad(self, D1, D2):
        xx,yy = np.meshgrid(D1.shape[0], D1.shape[0])
        D_magn = np.sqrt(D1**2 + D2**2)
        plt.quiver(xx, yy, D1, D2, D_magn)
        plt.draw()
        plt.pause(0.001)

        input("Press [enter] to continue.")
