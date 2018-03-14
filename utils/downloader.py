import urllib.request
import re
def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)
def get_google_file(google_id):
	url = 'https://drive.google.com/uc?export=download&id={}'.format(google_id)
	filename = '.data_cache/'+get_valid_filename(url)
	urllib.request.urlretrieve(url, filename)
	return filename