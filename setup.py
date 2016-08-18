from setuptools import setup, find_packages

setup(
    name='hands_on_tensorflow',
    version='0.1',
    description='Introduction to Tensorflow',
    url="",
    author='Yoann Benoit',
    author_email='',
    license='new BSD',
    packages=find_packages(),
    install_requires=['tensorflow'],
    tests_require=['pytest', "unittest2"],
    scripts=[],
    py_modules=["hands_on_tensorflow"],
    include_package_data=True,
    zip_safe=False
)
