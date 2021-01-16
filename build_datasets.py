import os
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image as im
from osgeo import gdal, ogr

from image_utils import sliding_window, get_extent, normalize, apply_contrast

image_size = 512

parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, nargs='?',
                    help="Input image e.g image.tif")

parser.add_argument('--image_bands', type=int, nargs='?',
                    help="Get first bands e.g 3")

parser.add_argument('--labels_file', type=str, nargs='?',
                    help="Labels e.g labels.shp")
parser.add_argument('--labels_field', type=str, nargs='?',
                    help="Labels field e.g class_id")
parser.add_argument('--dataset', type=str, nargs='?', default='train',
                    help="Dataset e.g train or test")

args = parser.parse_args()

image_path = args.image_file
image_bands = args.image_bands
labels_path = args.labels_file
labels_field = args.labels_field

images_folder = 'data/{d}_images'.format(d=args.dataset)
labels_folder = 'data/{d}_label'.format(d=args.dataset)
images_df_path = 'data/{d}.csv'.format(d=args.dataset)

image_dataset = gdal.Open(image_path)
labels_dataset = ogr.Open(labels_path)
image_transform = image_dataset.GetGeoTransform()

spatial_resolution_x = abs(image_transform[1])
spatial_resolution_y = abs(image_transform[5])
image_channels = image_bands or image_dataset.RasterCount

print(spatial_resolution_x, spatial_resolution_y, image_channels)

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

layer = labels_dataset.GetLayer(0)

image = image_dataset.ReadAsArray()
image = image.transpose(1, 2, 0)
extent = get_extent(image_dataset)


if os.path.exists(images_df_path):
    image_df = pd.read_csv(images_df_path)
    id = max(image_df['id'].tolist())
else:
    image_df = pd.DataFrame(columns=['id', 'image'])
    id = 0

filename, extension = os.path.basename(image_path).split('.')

filename = filename.replace('_', '-')

for (x, y, window) in sliding_window(image, image_size):
    chip = np.array(window[:, :, : image_channels])

    chip = np.flipud(chip)
    chip = apply_contrast(chip)
    chip = normalize(chip)
    chip = chip * 255

    left = extent[0] + (x * spatial_resolution_x)
    right = left + (image_size * spatial_resolution_x)
    top = extent[3] - (y * spatial_resolution_y)
    bottom = top - (image_size * spatial_resolution_y)

    chip_extent = [left, right, bottom, top]

    wkt = "POLYGON (({left} {bottom}," \
          "{left} {top}," \
          "{right} {top}," \
          "{right} {bottom}," \
          "{left} {bottom}))" \
        .format(left=left, right=right, top=top, bottom=bottom)

    layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))

    print("WKT: ", wkt)
    print("Count: {}".format(layer.GetFeatureCount()))

    if layer.GetFeatureCount() == 0:
        continue

    id += 1

    image_id = "{id}.tif".format(id=id)

    output_image = '{folder}/{filename}'.format(folder=images_folder,
                                                filename=image_id)

    output_labels = '{folder}/{filename}'.format(folder=labels_folder,
                                                filename=image_id)

    image_ds = gdal.GetDriverByName('GTiff').Create(output_image,
                                                     image_size,
                                                     image_size,
                                                     image_channels,
                                                     gdal.GDT_Byte,
                                                     ['COMPRESS=DEFLATE'])

    image_ds.SetProjection(image_dataset.GetProjection())

    image_ds.SetGeoTransform((left, spatial_resolution_x, 0,
                               bottom, 0, spatial_resolution_y))

    for band_idx in range(chip.shape[2]):
        band_data = chip[:,:,band_idx]
        band = image_ds.GetRasterBand(band_idx + 1)
        band.WriteArray(band_data)


    opts = ['TILED=YES', 'COMPRESS=DEFLATE', 'PREDICTOR=2']

    target_ds = gdal.GetDriverByName('GTiff').Create(output_labels,
                                                     image_size,
                                                     image_size,
                                                     1,
                                                     gdal.GDT_Byte,
                                                     opts)

    target_ds.SetProjection(image_dataset.GetProjection())

    target_ds.SetGeoTransform((left, spatial_resolution_x, 0,
                               bottom, 0, spatial_resolution_y))

    gdal.RasterizeLayer(target_ds, [1], layer, None, None, [1],
        ['ATTRIBUTE=%s' % labels_field, 'ALL_TOUCHED=TRUE'])

    target_ds = None
    out_datasource = None
    out_layer = None

    image_df = image_df.append({'id': id, 'image': image_id}, ignore_index=True)
    image_df.to_csv(images_df_path, index=None)
