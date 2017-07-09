from setuptools import setup

setup(name='dic',
      version='0.1',
      description='Optimizing experimental parameters for OI-DIC',
      url='https://github.com/scott-trinkle/DIC',
      author='Scott Trinkle',
      author_email='tscott.trinkle@gmail.com',
      license='MIT',
      packages=['dic'],
      install_requires=[
          'matplotlib',
          'numpy',
          'sympy'
      ],
      zip_safe=False)
