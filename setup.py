from setuptools import setup, find_packages

setup(name='marl-predator-prey',
      version='0.1',
      description='Distributed RL training application',
      url='#',
      author='Luca Fabri',
      author_email='luca.fabri@studio.unibo.it',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      test_suite='src.test',
      zip_safe=False
      )
