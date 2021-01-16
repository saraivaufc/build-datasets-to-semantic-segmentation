import numpy as np
from skimage import exposure

def get_extent(dataset):
    # Get extent
    geoTransform = dataset.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * dataset.RasterXSize
    miny = maxy + geoTransform[5] * dataset.RasterYSize

    # The extent kwarg controls the bounding box in data coordinates
    # that the image will fill specified as (left, right, bottom, top)
    extent = [minx, maxx, miny, maxy]
    return extent

def normalize(image):
    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized

def apply_contrast(image):
    image_rescaled = np.zeros(image.shape)
    for band_idx in range(image.shape[2]):
        in_low, in_high = 2, 98

        band_data = image[:, :, band_idx]
        valid_data = band_data[np.isfinite(band_data)]

        low, high = np.percentile(valid_data, (in_low, in_high))

        band_data = exposure.rescale_intensity(band_data,
                                               in_range=(low, high))

        image_rescaled[:, :, band_idx] = band_data
    return image_rescaled

def sliding_window(image, chip_size):

    chip_size_cols = chip_size
    chip_size_rows = chip_size

    step_cols = chip_size
    step_rows = chip_size

    cols = image.shape[1]
    rows = image.shape[0]

    for y in range(0, rows, step_rows):
        for x in range(0, cols, step_cols):
            origin_x = x
            origin_y = y

            if (origin_y + chip_size_rows) > rows:
                origin_y = rows - chip_size_rows

            if (origin_x + chip_size_cols) > cols:
                origin_x = cols - chip_size_cols

            chip = image[origin_y:origin_y + chip_size_rows,
                   origin_x: origin_x + chip_size_cols]

            yield (origin_x, origin_y, chip)
