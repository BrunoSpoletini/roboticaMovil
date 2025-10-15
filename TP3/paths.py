import os

if os.path.exists('/home/sco/roboticaMovil'):
    folder_path = '/home/sco/roboticaMovil'
elif os.path.exists('/home/bruno/roboticaMovil'):
    folder_path = '/home/bruno/roboticaMovil'
else: 
    folder_path = '..'
