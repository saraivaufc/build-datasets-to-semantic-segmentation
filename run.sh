RASTERS_DIR="rasters"
VECTORIAL_FILE="labels.gpkg"

for input in ${RASTERS_DIR}/*.tif; do
	echo ${input}
	python3 build_datasets.py --dataset=train --image_file=${input} --image_bands=3 --labels_file=${VECTORIAL_FILE} --labels_field=Crop2014_id
done

