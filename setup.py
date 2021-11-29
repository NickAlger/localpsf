from distutils.core import setup

setup(
    name='localpsf',
    version='0.1dev',
    packages=['localpsf',],
    license='MIT',
    author='Nick Alger',
    # long_description=open('README.txt').read(),
    include_package_data=True,
    package_data={'localpsf': ['angel_peak_badlands.png']},
)