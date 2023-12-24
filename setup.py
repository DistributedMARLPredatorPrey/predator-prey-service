from setuptools import setup, find_packages

setup(
    name="predator-prey-app",
    version="0.1.0",
    description="Distributed RL training application",
    url="#",
    author="Luca Fabri",
    author_email="luca.fabri1999@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="src.test",
    zip_safe=False,
)
