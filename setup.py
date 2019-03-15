from distutils.core import setup
#import distutils.command.bdist_conda

setup(
    name='slideslicer',
    version='0.1',
    packages=['slideslicer',],
    license='BSDv2',
    author="Dmytro S Lituiev",
    author_email="d.lituiev@gmail.com",
    description="tools for reading Leica digital slides and annotations, and cutting them in smaller patches",
    long_description=open('README.md').read(),
    setup_requires = ['cython','matplotlib','numpy',
                      ],
    install_requires=['pandas', 'openslide_python', 'Pillow>=5.0.0', 'opencv-python',
                      'scikit-image>=0.14.1', # used solely in hsv_histeq.py
                      'pycocotools',
                      'shapely', 'descartes', 
                      'lxml',
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSDv2",
        "Operating System :: OS Independent",
    ],
    #distclass=distutils.command.bdist_conda.CondaDistribution,
    conda_buildnum=1,
)
