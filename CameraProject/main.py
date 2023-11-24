# Things to pip install
# pip install numpy matplotlib jupyter torch opencv-python scikit-image facenet-pytorch jupyterthemes

#steps to make work
# first pip install in IDE terminal
# next, in IDE terminal go into camera module and pip install -e .   (note, make sure to use the ".")
# then use the jupyter notebook inside camera module to see if everything works
# now in IDE terminal go out and go into facenet-models and pip install -e .
# then use the jupyter notebook inside facenet_module module to see if everything works
# now go back into the larger project (cameraproject) and pip install -e .
# the facerec jupyter notebook will take and label photos, basics are set up, but saving to database and the like need to work on
# the whispers jupyter notebook allows you to take the database and determine their uniqueness. Work in Progress.



# if needed to run in terminal due to camera access in windows .\CameraProject\venv\Scripts\activate

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
