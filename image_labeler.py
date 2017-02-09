import shutil
import requests
import subprocess
import random
import os
import tensorflow as tf, sys
import datetime

class ImageLabeler:
    IMAGE_DIR = 'temp/images'

    def __init__(self):
        pass

    def download_image(self, image_url):
        """Downloads an image and returns a status"""
        image_file_name = hash(image_url)
        local_target_path = "{}/{}.jpg".format(self.IMAGE_DIR, image_file_name)

        if os.path.exists(local_target_path):
            return local_target_path

        image_request = requests.get(image_url, stream=True)

        if image_request.status_code == 200:
            with open(local_target_path, 'wb') as f:
                image_request.raw.decode_content = True
                shutil.copyfileobj(image_request.raw, f)
        else:
            return 'fail'

        return local_target_path

    def get_tags(self, image_url):
        image_path = self.download_image(image_url)

        # Show image within terminal using imgcat
        # Remove for external uses.
        # subprocess.Popen(['/bin/zsh', '-i', '-c', 'imgcat {}'.format(image_path)])

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
                           in tf.gfile.GFile("tf_files_1/retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("tf_files_1/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        scores = {}

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            sigmoid_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(sigmoid_tensor, {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                scores["_".join(human_string.split())] = score

        return scores
