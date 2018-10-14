import os
import urllib.request

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_image_from_url(path, image_url):
    """
    Save file from url
    :param path: where the file will be save
    :param image_url: string url
    :return:
    """
    filename = image_url.split('/')[-1]
    urllib.request.urlretrieve(image_url, os.path.join(path, filename))
    return filename


def save_upload_file(path, file):
    """
    Save file
    :param path: where the file will be save
    :param file: request.files['image']
    """
    filename = file.filename
    if file and allowed_file(filename):
        file.save(os.path.join(path, filename))
        return filename
    else:
        return ''
