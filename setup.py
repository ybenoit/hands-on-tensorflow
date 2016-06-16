from setuptools import setup, find_packages

setup(
    name='tutorials',
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
    py_modules=["tutorials"],
    include_package_data=True,
    zip_safe=False
)
