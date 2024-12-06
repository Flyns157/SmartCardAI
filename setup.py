import setuptools

def _get_version(path: str) -> float | str:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f'''{path} doesn't exist !''')
    
    with open(os.path.join(path, '__init__.py') if os.path.isdir(path) else path) as f:
        for line in f:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version__']
        raise ValueError('`__version__` not defined')

VERSION = _get_version('rlcard_trainer')

setuptools.setup(
    name="rlcard-trainer",
    version=VERSION,
    author="CUISSET Mattéo and COPIN Lucas",
    author_email="matteo.cuisset@gmail.com",
    description="A Toolkit for Reinforcement Learning in Card Games",
    long_description=open("README.md", "r", encoding="utf8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Flyns157/SmartCardAI",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "Manager", "Management"],
    packages=setuptools.find_packages(exclude=('tests',)),
    package_data={
        'rlcard_trainer': [
            'pretrained/*',
        ]},
    install_requires=open('requirements.txt').read().splitlines(),
    requires_python='>=3.12',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
