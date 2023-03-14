from math import log10, sqrt
import cv2
import numpy as np
a = 0.4
m = 5


def PSNR(original, compressed):

    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def filled(image, res):

    ins = [i*(m) for i in range(1, res+1)]

    idal_shape = np.zeros((image.shape[0], image.shape[1]+res-1))
    img = np.insert(image, ins, [[y]
                    for y in np.zeros(image.shape[0])], axis=1)
    # print (img.shape)
    # print(idal_shape.shape)
    # print(image.shape)
    for j in range(0, (idal_shape.shape[1])):
        if j % (m) == 0:

            idal_shape[:, j] = np.rint(np.sum(a*img[:, j-m:j], axis=1))

        else:
            idal_shape[:, j] = img[:, j]

    return idal_shape


def pre_decode(image):

    blue, green, red = cv2.split(image)
    temp = np.array([0, 0, 0], dtype=object)
    res = int(image.shape[1]/m)

    temp[0] = filled(blue, res)
    temp[1] = filled(green, res)
    temp[2] = filled(red, res)

    final = cv2.merge([temp[0], temp[1], temp[2]])

    # print (final[:,4,:])
    # print(PSNR(image,final))
    filename = 'decodepref_4.bmp'

    cv2.imwrite(filename, final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


def pre_encode(img):

    blue, green, red = cv2.split(img)
    del_list = []
    for j in range(m, (img.shape[1])-1):
        if j % (m+1) == 0:
            del_list.append(j)
    encode_img_b = np.delete(blue, del_list, 1)
    encode_img_g = np.delete(green, del_list, 1)
    encode_img_r = np.delete(red, del_list, 1)

    final = cv2.merge([encode_img_b, encode_img_g, encode_img_r])
    filename = 'encodpref_4.bmp'
    cv2.imwrite(filename, final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


if __name__ == '__main__':
    img = cv2.imread("/home/atefeh/Desktop/image/compression/traf.bmp")
    encode = pre_encode(img)
    final = pre_decode(encode)
    print(PSNR(img, final))
    print(f"encode size {encode.size}")
    import matplotlib.pyplot as plt

    from PIL import Image
    imgplot = plt.imshow(final)
    plt.savefig("treebit.png")