# build-datasets-to-semantic-segmentation

### Requirements
* GDAL -- [Homepage](http://www.gdal.org)

### Install Gdal
Get gdal development libraries:
```shell
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python3-dev
$ sudo apt-get install gdal-bin python3-gdal
```


### Install virtual enviroment

```shell
$ sudo apt-get install virtualenv
```

### Create and activate a virtual environment
```shell
$ virtualenv env -p python3
$ source env/bin/activate
```

Install Others Requirements
```shell
(env) $ pip3 install -r requirements.txt
```

### Install GDAL
```shell
(env) $ pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```

### Usage

```shell
(env) $ python3 build_datasets.py --dataset=train --image_file=image.tif --image_bands=3 --labels_file=labels.gpkg --labels_field=Crop2014_id
```

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
    <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" />
</a>
<br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
