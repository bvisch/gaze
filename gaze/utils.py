def find_pupil(eye):
    # smooth eye to dull specular lighting
    eye = cv2.GaussianBlur(eye,(5,5),0.01*eye.shape[1])

    # calculate all distance vectors
    eye_len = np.arange(eye.shape[0])
    xx,yy = np.meshgrid(eye_len,eye_len)
    X1,X2 = np.meshgrid(xx.ravel(),xx.ravel()) # [1,2,3,1,2,3,1,2,3] & [1,1,1,1,1,1,1,1,1] ([2,2,2,2,2,2,2,2,2] would be next)
    Y1,Y2 = np.meshgrid(yy.ravel(),yy.ravel()) # [1,1,1,2,2,2,3,3,3] & [1,1,1,1,1,1,1,1,1] ([1,1,1,1,1,1,1,1,1] would be next)
    Dx,Dy = [X1-X2,Y1-Y2]
    Dlen = np.sqrt(Dx**2+Dy**2)
    Dx,Dy = [Dx/Dlen,Dy/Dlen] #normalized

    # get gradient
    Gx,Gy = np.gradient(eye.astype('float32'))
    Gmagn = np.sqrt(Gx**2+Gy**2)
    Gx,Gy = [Gx/Gmagn,Gy/Gmagn] #normalized
    GX,GY = np.meshgrid(Gx.ravel(),Gy.ravel())

    X = (GX*Dx+GY*Dy)
    #X[X<0] = 0
    X = X**2

    # weight darker areas higher by multiplying by inverted image
    eye = cv2.bitwise_not(eye)
    eyem = np.repeat(eye.ravel()[np.newaxis,:],eye.size,0)
    C = (np.nansum(eyem*X, axis=0)/eye.size).reshape(eye.shape)

    mask = np.ones(C.shape, dtype=bool)
    mask[1:-1,1:-1] = False
    C[mask] = 0
    #threshold = 0.9*C.max()
    #C[C<threshold] = 0
    #retval,C = cv2.threshold(C.astype(np.float32), threshold, 0, cv2.THRESH_TOZERO)

    return np.unravel_index(C.argmax(), C.shape)[::-1]


def gradient_ascent(alpha, c, eye, g):
    eye_len = np.arange(eye.shape[0])
    xx,yy = np.meshgrid(eye_len, eye_len)
    xx,yy = [xx-c[0],yy-c[1]]
    n = np.sqrt(xx**2 + yy**2)

    gx,gy = np.gradient(eye.astype('float32'))
    e = (xx*gx + yy*gy)

    D1 = ((xx-c[0]) * e**2 - gx * e * n**2) / n**4
    D2 = ((yy-c[1]) * e**2 - gy * e * n**2) / n**4

    D1 = 2/D1.length * np.nansum(D1)
    D2 = 2/D2.length * np.nansum(D2)

    return [c[0] + alpha*D1,c[1] + alpha*D2]


def find_pupil(eye):


    dc1 = 2/eye.length *
    dc2
