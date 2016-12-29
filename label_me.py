from collections import defaultdict
from os import path
from random import shuffle
from requests import get
from shutil import copyfileobj
from lxml import html

class LabelMe:
    """
    Simple API interface to download images from LabelMe's image database, a project that provides
    digital images and annotations. Basic directory search is also supported.
    For more info, visit http://labelme2.csail.mit.edu/Release3.0/index.php.
    """

    BASE_IMAGE_DIR_URL = 'http://people.csail.mit.edu/brussell/research/LabelMe/Images/'
    VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png']

    def __init__(self, tree=None):
        """Initialize with an existing tree or get the source tree from LabelMe's site"""
        self.image_html_tree = tree or self.get_tree_from_source(self.BASE_IMAGE_DIR_URL)
        self.cached_search_stats = {}

    def get_tree_from_source(self, dir_url):
        """Retrieves the parsed HTML tree from a given directory URL."""
        dir_request = requests.get(dir_url)
        return html.fromstring(dir_request.content)

    def search_links(self, search_term):
        """Returns the matching search links for a given search term """
        # Returns an array of "Element a" objects.
        links_results = self.image_html_tree.xpath('.//a[contains(text(), "{}")]'.format(search_term))

        # Only get the links from the results
        return random.shuffle(
          [link_result.get('href') for link_result in links_results if self.valid_dir(link_result.get('href'))]
        )

    def search_stats(self, search_term):
        """Returns stats for a given search term including total number of matching dirs & images."""
        if search_term not in self.cached_search_stats.keys():
            search_stats = defaultdict(int)
            found_dir_links = self.search_links(search_term)
            search_stats['number_of_matching_dirs'] += len(found_dir_links)

            for dir_link in found_dir_links:
                images_dir_path = self.BASE_IMAGE_DIR_URL + dir_link
                image_link_elements = self.get_tree_from_source(images_dir_path).xpath('.//a')
                image_urls = [element.get('href') for element in image_link_elements if self.valid_image(element.get('href'))]
                search_stats['total_number_of_images'] += len(image_urls)
                search_stats[dir_link] += len(image_urls)

            self.cached_search_stats[search_term] = dict(search_stats)

        print(self.cached_search_stats[search_term])

    def download_images_from_dir(self, dir_path, user_path=None):
        """Downloads all valid images from a given directory to a local directory and outputs process stats"""
        # Downloads the images into a local directory with the same dir_path name or with a provided dir_path.
        images_dir_path = self.BASE_IMAGE_DIR_URL + dir_path
        image_link_elements = self.get_tree_from_source(images_dir_path).xpath('.//a')

        # Only gets the valid image links from within this directory
        image_urls = [element.get('href') for element in image_link_elements if self.valid_image(element.get('href'))]
        self.maybe_create_directory((user_path or ('images/' + dir_path)))

        download_statuses = defaultdict(int)

        for image_url in image_urls:
            download_statuses[self.download_image(dir_path, image_url, user_path)] += 1

        print("Downloaded: {}, Skipped: {}, Failed: {}".format(download_statuses['success'], download_statuses['skip'], download_statuses['fail']))
        return download_statuses

    def download_image(self, target_dir, image_file, user_path=None):
        """Downloads an image and returns a status"""
        image_url_source = self.BASE_IMAGE_DIR_URL + target_dir + image_file
        local_target_path = (user_path or ('images/' + target_dir)) + image_file

        if os.path.isfile(local_target_path):
            print("Duplicate file detected, skipping image file: {}".format(image_file))
            return 'skip'

        image_request = requests.get(image_url_source, stream=True)
        if image_request.status_code == 200:
            with open(local_target_path, 'wb') as f:
                image_request.raw.decode_content = True
                shutil.copyfileobj(image_request.raw, f)
            return 'success'
        else:
            return 'fail'

    def valid_image(self, url):
        """Checks for a valid extension for a given image URL and is not part of a movie file"""
        return url.endswith(tuple(self.VALID_IMAGE_EXT))

    def valid_dir(self, dir_path):
        """Exclude repetitive movie or sequential frame dirs"""
        return all(inv not in dir_path for inv in ['mvi', 'seq'])

    def maybe_create_directory(self, dir_name):
        """Creates a directory if one does not previously exist"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
