import numpy as np
from PIL import Image, ImageDraw


def make_phantom(rows, cols):
    '''This function generates a phantom for use in testing the propagation 
    of variance in gamma and theta into the OPL

    Parameters
    __________
    rows : int
        Number of rows in the image
    cols : int
        Number of columns in the image

    Returns
    _______
    OPL : ndarray
        Phantom image as ndarray
    '''

    OPL = 50 * np.ones((rows, cols))
    OPL = Image.fromarray(OPL)

    draw = ImageDraw.Draw(OPL)

    # Rectangle 1
    x0 = cols * 0.045
    x1 = cols * 0.25
    y0 = rows * 0.08
    y1 = rows * 0.45
    draw.rectangle([x0, y0, x1, y1], fill=50.1)

    # Rectangle 2
    x0 = cols * 0.10
    x1 = cols * 0.2
    y0 = rows * 0.15
    y1 = rows * 0.40
    draw.rectangle([x0, y0, x1, y1], fill=49.8)

    # Rectangle 3
    x0 = cols * 0.15
    x1 = cols * 0.45
    y0 = rows * 0.6
    y1 = rows * 0.9
    draw.rectangle([x0, y0, x1, y1], fill=50.1)

    # Circle 1
    x0 = cols * 0.70
    x1 = cols * 0.75
    y0 = rows * 0.60
    y1 = rows * 0.65
    draw.ellipse([x0, y0, x1, y1], fill=50.2)

    # Circle 2
    x0 = cols * 0.80
    x1 = cols * 0.90
    y0 = rows * 0.50
    y1 = rows * 0.60
    draw.ellipse([x0, y0, x1, y1], fill=50.3)

    # Circle 3
    x0 = cols * 0.30
    x1 = cols * 0.55
    y0 = rows * 0.70
    y1 = rows * 0.95
    draw.ellipse([x0, y0, x1, y1], fill=50.2)

    # Ellipse 1
    x0 = cols * 0.45
    x1 = cols * 0.75
    y0 = rows * 0.25
    y1 = rows * 0.45
    draw.ellipse([x0, y0, x1, y1], fill=49.7)

    OPL = np.array(OPL)
    return OPL
