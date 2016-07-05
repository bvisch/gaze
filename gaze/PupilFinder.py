import numpy as np
from math import sqrt

class PupilFinder:

    def __init__(self, alpha = 0.05, max_iterations = 100):
        self.alpha = alpha
        self.max_iterations = max_iterations

    def gradient_ascent(self, eye, c, gx, gy):
        eye_len = np.arange(eye.shape[0])
        xx,yy = np.meshgrid(eye_len, eye_len)
        xx,yy = [xx-c[0],yy-c[1]]
        n = np.sqrt(xx**2 + yy**2)

        e = (xx*gx + yy*gy)

        D1 = ((xx-c[0]) * e**2 - gx * e * n**2) / n**4
        D2 = ((yy-c[1]) * e**2 - gy * e * n**2) / n**4

        D1 = 2/float(D1.size) * np.nansum(D1)
        D2 = 2/float(D2.size) * np.nansum(D2)

        last_difference = sqrt((self.alpha*D1)**2+(self.alpha*D2)**2)
        return (last_difference, [c[0] + self.alpha*D1,c[1] + self.alpha*D2])


    def find_pupil(self, eye):
        gx,gy = np.gradient(eye.astype('float32'))
        g_magn = np.sqrt(gx**2+gy**2)
        c_pos = np.argmax(g_magn)
        c = [gx.flat[c_pos],gy.flat[c_pos]]
        last_difference = float('inf')
        num_iterations = 0

        while (num_iterations < self.max_iterations) and (last_difference > 0.03):
            last_difference,c = self.gradient_ascent(eye, c, gx, gy)
            num_iterations += 1

        return c
