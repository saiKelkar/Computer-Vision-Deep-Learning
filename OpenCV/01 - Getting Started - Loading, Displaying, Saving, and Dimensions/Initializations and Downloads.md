```
# Import OpenCV
import cv2

# Version of CV we are running
print(cv2.__version__) # 4.10.0

# Import other Libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# ZipFile module provides tools to create, read, write, append, and list a ZIP file
from zipfile import ZipFile

# urllib.request function helps in opening URLs (mostly HTTPS)
from urllib.request import urlretrieve

# To display images directly inside Jupyter Notebook
from IPython.display import Image
%matplotlib inline
```

```
# Download Assets
def download_and_unzip(url, save_path):
	print(f"Downloading and extracting assets...", end = "")

	# Downloading zip file using urllib package
	urlretrieve(url, save_path)

	try:
		# Extracting zip file using ZipFile package
		with ZipFile(save_path) as z:
			# Extract ZIP file contents in the same directory
			# .split, splits the path into (directory, filename)
			# example:
			# os.path.split("home/user/files/myfile.zip")
			# Output: ('/home/user/files', 'myfile.zip')
			z.extractall(os.path.split(save_path)[0])
		print("Done")

	except Exception as e:
		print("\nInvalid file", e)
```

"with" statement in Python:

```
with open("example.txt", "r") as file:
	content = file.read()
print(content)
```

"with" statement is used to manage resources like files, network connections, etc., efficiently. 
It ensures that the resources is automatically closed when you're done with it.

In above example: 
- open("example.txt", "r") opens the file
- assigns it to the variable "file"
- reads the content within the "with" block
- the file is closed once we exist the block (no need to call "file.close()" )

"save_path" in Python:

"save_path" is a regular Python variable, not a built-in function
"save_path" should be defined before this block is executed

example:

```
save_path = "path/to/my_file.zip"
```

if it is undefined, Python raises an error:

```
NameError: name 'save_path' is not defined
```

Accessing the function:

```
# r before the string makes it a raw string - useful if the URL has special characters like \, which Python might otherwise interpret as an escape sequence
URL = r"https://url_to_download_zip_file"

# os.getcwd() <-- returns the current working directory (where the script is running)
# os.path.join() <-- joins directory paths in an OS-independent way (works on Windows, Linux, macOS)
# example:
# os.path.join("/home/user", "file.zip")
# Output: "/home/user/file.zip"
# asset_zip_path <-- Full path to the ZIP file
asset_zip_path = os.path.join(os.getcwd(), "file_name.zip")

# Download if asset ZIP does not exists
if not os.path.exists(asset_zip_path):
	download_and_unzip(URL, asset_zip_path)
	# Downloading and extracting assets... Done
```

Downloading and zip extraction process (explained):
- define the URL (where the zip file is located) and Asset path (full path to save the zip file in your current working directory)
- check if the zip file exists 
  - if it doesn't, we proceed to download and extract it
  - if it already exists, we skip downloading to save time
- call download_and_unzip()
  - function downloads the zip from the provided url
  - saves it at the asset_zip_path
  - extracts the content of the zip file
- inside the download_and_unzip():
  - urlretrieve downloads the zip file from the url and saves it to the save_path
  - opens the zip file
  - extracts the contents of the zip file into the target directory, i.e., the target path "0" that we've mentioned in the code
  - if something goes wrong, an error is printed
