
import numpy as np
import cv2
import math
from scipy.ndimage.filters import convolve


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208856237

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    signal_len, kernel_len = np.size(in_signal), np.size(k_size)
    conv1_len = signal_len + kernel_len - 1
    result = np.zeros(conv1_len)

    p_signal = np.pad(in_signal, (kernel_len - 1, kernel_len - 1), 'constant')
    for i in np.arange(signal_len):
        for j in np.arange(kernel_len):
            result [i + j] = result[i + j] + (in_signal[i] * k_size[j])
            return result

def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :rtype: object
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    img_height, img_width = in_image.shape
    kernel_height, kernel_width = kernel.shape

    p_height = kernel_height // 2
    p_width = kernel_width // 2
    p_image = np.pad(in_image, ((p_height, p_height), (p_width, p_width)), 'median')

    convo_img= np.zeros_like(in_image)
    for i in range(img_height):
        for j in range(img_width):
            conv_pixel = p_image[i:i + kernel_height, j:j + kernel_width].sum()
            convo_img[i, j] = conv_pixel

    return convo_img


def convDerivative(in_image: np.ndarray) ->(np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    x_kernal = np.array([[1, 0, -1]])
    y_kernal = np.array([[0, 1, 0],
                        [0, 0, 0],
                       [0, -1, 0]])

    dx = conv2D(in_image, x_kernal)
    dy = conv2D(in_image, y_kernal)

    mag_image= np.sqrt(dx ** 2 + dy ** 2)
    direc_image = np.arctan2(dy, dx)

    return mag_image,direc_image

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaus_1d = np.array([[np.exp(-(np.square(x - (k_size - 1) / 2)) / (2 * np.square( 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8)))
                         for x in range(k_size)]])
    gaus_1d /= gaus_1d.sum()
    result= np.transpose(gaus_1d)

    img_result=conv2D(in_image, np.dot(gaus_1d, np.transpose(gaus_1d)))

    return img_result

def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = Gaussian_filter(k_size, -1)
    blur_image2 = cv2.filter2D(in_image, -1, kernel)

    return blur_image2


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    return

def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    if img.ndim == 3:
        img = np.median(img, axis=1)

    kernel_size = 10
    sigma = 1.0
    kernel = LoGKernel(kernel_size, sigma)
    img_filtar = convolve(img, kernel)
    edges = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            situation = [img_filtar[i - 1, j], img_filtar[i + 1, j], img_filtar[i, j - 1], img_filtar[i, j + 1]]
            if np.prod(np.sign(situation)) < 0:
                edges[i, j] = 255

    return edges

def LoGKernel(k_size, sigma):
    kernel_LOG = np.zeros((k_size, k_size))
    half_size= k_size // 2

    for i in range(k_size):
        for j in range(k_size):
            x = i - half_size
            y = j - half_size
            kernel_LOG[i, j] = (x**2 + y**2 - 2 * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel_LOG = kernel_LOG / np.sum(kernel_LOG)

    return kernel_LOG

def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
        """
        Find Circles in an image using a Hough Transform algorithm extension :param I: Input image
        :param minRadius: Minimum circle radius
        :param maxRadius: Maximum circle radius
        :return: A list containing the detected circles,
        [(x,y,radius),(x,y,radius),...]
        """
        edge_Canny = cv2.Canny(img, 300, 5)
        circle_shape = cv2.HoughCircles(edge_Canny, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                   param1=80, param2=60, minRadius=min_radius, maxRadius=max_radius)
        circles_hough = []
        if circle_shape is not None:
            circle_shape = np.round(circle_shape[0, ])

            circles_hough.extend([(x, y, r)
                                     for x, y, r
                                  in circle_shape])
        return circles_hough

def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    height, width = in_image.shape[:2]

    cv2_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    output_image = np.empty_like(in_image)
    kernel_size= k_size // 2
    img_pad = cv2.copyMakeBorder(in_image, kernel_size, kernel_size,kernel_size, kernel_size, cv2.BORDER_REPLICATE)

    result1=img_pad.shape[0] -kernel_size
    result2=img_pad.shape[1] - kernel_size
    for y in range(kernel_size, result1):
        for x in range(kernel_size, result2):

            pixol = img_pad[y, x]
            situation = img_pad[y - kernel_size: y + kernel_size + 1,x - kernel_size: x + kernel_size + 1]

            distances = pixol - situation
            W_P= np.exp(-0.5 * np.power(distances / sigma_color, 2))
            W_S = Gaussian_filter(2 * kernel_size + 1, sigma_space)
            result_Out = W_S * W_P
        output_image[y - kernel_size, x - kernel_size] = np.sum(result_Out * situation) / np.sum(result_Out)

    return cv2_image, output_image


def Gaussian_filter(k, sigma):
       result1=math.exp(-0.5 * (k**2) / (sigma**2))
       result2=result1/(math.sqrt(2 * math.pi) * sigma)
       return result2














