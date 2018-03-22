To begin training:

1) Clone the full TensorFlow repo from Google
2) Open it, then navigate to Models\research
3) Run:

python3 object_detection/train.py --logtostderr --pipeline_config_path=C:\{path to FDDB_SSD_mobilenet.config} --train_dir=C:\{path to halfbaked folder}


To fully bake (freeze, export, whatever):

python3 object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=C:\{path to FDDB_SSD_mobilenet.config} --trained_checkpoint_prefix=C:\{path to halfbaked\model.ckpt-xxx} --output_directory=C:\{path to fullybaked folder}

**xxx represents the checkpoint number
**make sure to delete the existing contents of fullybaked before re-baking