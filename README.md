# Video Steganography

[![Build Status](https://travis-ci.com/Amritaryal44/drawing-with-DFT-and-epicycle.svg?token=pZnXxmooEKHShwNV59AQ&branch=master)](https://travis-ci.com/Amritaryal44/drawing-with-DFT-and-epicycle)

Draw any figure using maths with the use of DFT(fouries series in exponential form) and epicycle
### Example
| real figure |
|:--:|
| <img src="nepal.png" width="50%" height="50%"> | 

| generated figure |
|:--:|
| ![Generated figure](img_md/example.gif) |


### Installation

```sh
$ sudo apt-get update
$ sudo apt-get install ffmpeg
```

Clone the repository
```sh
$ git clone https://github.com/Amritaryal44/Video-Steganography.git
```
Install all package requirements for python
```sh
$ pip install -r requirements.txt
```

**To hide the video, use encrypt.py:**
```python
# path of cover video and secret video
cover_path = "v1.mp4"
secret_path = "v2.mp4"
```
- ```cover_path``` : Stores the path of cover video in which secret video is to be hidden.
- ```secret_path``` : Stores the path of secret video which is to be hidden.

Encrypted video is saved in ```out/covered.mkv```

**To reveal the secret video, use decrypt.py:**
```python
# path to encrypted file
enc_path = "out/covered.mkv"
```
- ```cover_path``` : Stores the path of cover video in which secret video is to be hidden.
- ```secret_path``` : Stores the path of secret video which is to be hidden.

Decrypted video is saved in ```out/secret_revealed.mkv```

