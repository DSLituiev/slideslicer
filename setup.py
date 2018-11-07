from distutils.core import setup

setup(
    name='slideslicer',
    version='0.1',
    packages=['slideslicer',],
    license='BSDv2',
    author="Dmytro S Lituiev",
    author_email="d.lituiev@gmail.com",
    description="tools for reading Leica digital slides and annotations, and cutting them in smaller patches",
    long_description=open('README.md').read(),
    install_requires=['openslide_python', 'Pillow>=5.0.0', 'pycocotools',
                      'shapely', 'opencv-python', 'beautifulsoup4>=4.6.0',
                      'scikit-image>=0.14.1', # used solely in hsv_histeq.py
                      'descartes',
                      'matplotlib','numpy','pandas', 'lxml'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSDv2",
        "Operating System :: OS Independent",
    ],
)
