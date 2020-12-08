#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'MultiSim'
DESCRIPTION = 'Thermal simulation tool for heating appliances.'
URL = 'https://github.com/JoElfner/multisim'
download_url = 'https://github.com/JoElfner/multisim/archive/v{0}.tar.gz'
EMAIL = 'johannes.elfner@hm.edu'
AUTHOR = 'Johannes Elfner'
REQUIRES_PYTHON = '>=3.7.9'
VERSION = '0.10.0'

download_url = download_url.format(VERSION)

# What packages are required for this module to be executed?
REQUIRED = [
    'matplotlib (>=3.3.2)',
    'numba (>=0.51.2)',
    'numpy (>=1.19.2)',
    'pandas (>=1.0.5)',
    'scipy (>=1.5)',
    'scikit-learn (>=0.23.1)',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}


def write_version_py(version, filename='multisim/version.py'):
    """Write version to file for module wide access."""
    cnt = """# THIS FILE IS GENERATED FROM MultiSim SETUP.PY

version = '%(version)s'
"""
    with open(filename, 'w') as f:
        f.write(cnt % {'version': version})


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for it!

base_dir = os.path.abspath(os.path.dirname(__file__))
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# same with changelog:
try:
    with open(os.path.join(base_dir, "CHANGELOG.rst")) as f:
        # Remove :issue:`ddd` tags that breaks the description rendering
        changelog = f.read()
except FileNotFoundError:
    changelog = 'changelog not found'
long_description = "\n\n".join([long_description, changelog])


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(base_dir, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

write_version_py(about['__version__'])


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(base_dir, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --python-tag py{1}'.format(
                sys.executable, REQUIRES_PYTHON.replace('.', '')[-3:-1]
            )
        )

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload --config-file .pypirc dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=download_url,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    # $ setup.py publish support.
    cmdclass={'upload': UploadCommand},
)
