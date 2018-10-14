# Simple image classifier serving

## Clone project
```shell
$ git clone
```

## Developer Install

### Install virtualenv

```shell
$ pip install virtualenv
```

### Create and active project virtualenv

```shell
$ virtualenv simple-image-classifier-serving
$ source simple-image-classifier-serving/venv/bin/activate
```

### Install the dependencies

```shell
$ pip -r requirements.txt
```

## Retrain

### Init data set

**In theory all you'll need to do is point it at a set of sub-folders, each named after one of your categories and containing only images from that category.**

```shell
$ curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
$ tar xzf flower_photos.tgz -C ./training_data
```

### Retrain

```shell
$ python scripts/retrain.py \
   --output_graph=tf_files/flowers_retrained_graph.pb \ # Your model file
   --output_labels=tf_files/flowers_labels.txt \ # Your lable file
   --image_dir=training_data/flower_photos \ # Your training data folder
   --how_many_training_steps=200 # Recommended default value 4000
```

### Test Your Image Classifier

```shell
python scripts/label_image.py \
   --graph=tf_files/flowers_retrained_graph.pb \
   --labels=tf_files/flowers_labels.txt \
   --input_layer=Placeholder \
   --output_layer=final_result \
   --image=training_data/flower_photos/roses/rose_test_1.jpg
```

## Serving
You maybe have to change value of 4 consts in head of app.py file
```python
LABELS_FILE = './tf_files/flowers_labels.txt'
MODEL_FILE = './tf_files/flowers_retrained_graph.pb'
UPLOAD_FOLDER = './temp'
PORT = 12480
```

Start server
```shell
$ python app.py
```

APIs testing:

* GET method

    ```shell
    curl -X GET 'http://127.0.0.1:12480/classify?image_url=https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Rose_Amber_Flush_20070601.jpg/270px-Rose_Amber_Flush_20070601.jpg'
    ```
    
    Result
    
    ```shell
    {
      "filename": "270px-Rose_Amber_Flush_20070601.jpg", 
      "labels": {
        "daisy": 0.005170978605747223, 
        "dandelion": 0.005906935781240463, 
        "roses": 0.8705077767372131, 
        "sunflowers": 0.029377983883023262, 
        "tulips": 0.0890362486243248
      }
    }
    ```
* POST method (file uploading)
    
    ```shell
    curl -X POST \
    'http://127.0.0.1:12480/classify' \
    -F "image=@./huong_duong_5.jpg"
    ```
    
    Result
    
    ```shell
    {
      "filename": "huong_duong_5.jpg", 
      "labels": {
        "daisy": 0.13612933456897736, 
        "dandelion": 0.04980277642607689, 
        "roses": 0.0031164991669356823, 
        "sunflowers": 0.8034898042678833, 
        "tulips": 0.007461594417691231
      }
    }
    ```
 