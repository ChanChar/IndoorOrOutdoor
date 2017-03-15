import shutil
import requests
import subprocess
import random
import os
import tensorflow as tf, sys
import datetime
import concurrent.futures

class ImageLabeler:
    IMAGE_DIR = 'temp/images'
    AVAIL_THREADS = 4

    def __init__(self):
        self.setup_graph()

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

    def setup_graph(self):
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files_1/retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("tf_files_1/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def get_tags(self, image_urls, concurrent=False):
        self.image_urls = image_urls
        self.process_images(image_urls)
        sess = tf.Session()
        scores = []

        if concurrent:
            scores = self.conn_get_tags(sess)
        else:
            for i, image_data in enumerate(self.image_data):
                scores.append(self.get_image_score((self.image_urls[i], image_data, sess)))
                print("Completed {}%".format((i+1) / len(self.image_urls) * 100))

        return scores

    # See if TF handles concurrent sessions.
    def conn_get_tags(self, session):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.AVAIL_THREADS) as executor:
            # Start the load operations and mark each future with its URL
            # image_paths = { executor.submit(self.download_image, image_url): image_url for image_url in image_urls }
            image_data = [ (self.image_urls[i], image_data, session) for i, image_data in enumerate(self.image_data)]
            # paths = executor.map(self.download_image, image_urls)
            image_scores = executor.map(self.get_image_score, image_data)

        return [score for score in image_scores]


    def get_image_score(self, image):
        # With the passed in session
        sess = image[2]
        with sess.as_default() as sess:
            # Feed the image_data as input to the graph and get first prediction
            sigmoid_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(sigmoid_tensor, {'DecodeJpeg/contents:0': image[1]})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            image_score = { 'image_url': image[0] }

            for node_id in top_k:
                image_score["_".join(self.label_lines[node_id].split())] = predictions[0][node_id]

            return image_score


    def process_images(self, image_urls):
        # download each image given the image url concurrently using threads (try processes next time?).
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            paths = executor.map(self.download_image, image_urls)

        self.image_paths = [path for path in paths]
        print("Processing {} images".format(len(self.image_paths)))
        # Read in the image_data
        self.image_data = [tf.gfile.FastGFile(image_path, 'rb').read() for image_path in self.image_paths]


    # need to update shell if using something other than zsh.
    # uses imgcat to display the image inline with the terminal.
    def display_in_terminal(self, image_path):
        subprocess.Popen(['/bin/zsh', '-i', '-c', 'imgcat {}'.format(image_path)])
