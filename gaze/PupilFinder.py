import numpy as np

class PupilFinder:

    def __init__(self, alpha = 0.05, max_iterations = 100):
        self.c = [0,0]
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.last_difference = float('inf')
        self.num_iterations = 0

    def gradient_ascent(self, eye):
        eye_len = np.arange(eye.shape[0])
        xx,yy = np.meshgrid(eye_len, eye_len)
        xx,yy = [xx-self.c[0],yy-self.c[1]]
        n = np.sqrt(xx**2 + yy**2)

        gx,gy = np.gradient(eye.astype('float32'))
        e = (xx*gx + yy*gy)

        D1 = ((xx-self.c[0]) * e**2 - gx * e * n**2) / n**4
        D2 = ((yy-self.c[1]) * e**2 - gy * e * n**2) / n**4

        D1 = 2/float(D1.size) * np.nansum(D1)
        D2 = 2/float(D2.size) * np.nansum(D2)

        self.last_difference = self.alpha*D1*self.alpha*D2
        self.c = [self.c[0] + self.alpha*D1,self.c[1] + self.alpha*D2]


    def find_pupil(self, eye):
        import pdb; pdb.set_trace()
        while (self.num_iterations < self.max_iterations) or (self.last_difference > 0.3):
            self.gradient_ascent(eye)
            self.num_iterations += 1

        return self.c
