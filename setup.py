from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='VAST',
    version='0.0.1',
    author='Jiayao Yan',
    author_email='jiy352@ucsd.edu',
    license='LGPLv3+',
    keywords='vortex-based aerodynamic solver toolkit',
    url='http://github.com/jiy352/vast',
    # download_url='http://pypi.python.org/pypi/lsdo_project_template',
    description='vortex-based aerodynamic solver toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        # 'numpy',
        # 'pytest',
        'myst-nb',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        # 'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
        'VLM_package @ git+https://github.com/jiy352/lsdo_VLM',
        'python_csdl_backend @ git+https://github.com/LSDOlab/python_csdl_backend',
        'csdl @ git+https://github.com/LSDOlab/csdl',  
        'csdl_om @ git+https://github.com/LSDOlab/csdl_om', 
        'zone @ git+https://github.com/LSDOlab/ozone.git@main#egg=ozone'

        # 'ozone @ git+https://github.com/LSDOlab/ozone', 
        # 'VAST @ git+https://github.com/jiy352/VAST',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
)
