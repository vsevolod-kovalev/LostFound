# Script to download relevant categories of lost/found items from Image-Net
import requests
from constants import image_net_synsets

for synset_id in image_net_synsets:
    url = f"https://www.image-net.org/data/winter21_whole/{synset_id}.tar"
    response = requests.get(url)
    with open(f"{synset_id}.tar", 'wb') as file:
        file.write(response.content)
