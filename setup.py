import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

extras = {
    'torch': ['torch', 'GitPython', 'gitdb2', 'matplotlib'],
}

def _get_version():
    with open('SmartCardAI/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version__']
        raise ValueError('`__version__` not defined')

VERSION = _get_version()

setuptools.setup(
    name="SmartCardAI",
    version=VERSION,
    author="CUISSET Mattéo and COPIN Lucas",
    author_email="matteo.cuisset@gmail.com",
    description="A Toolkit for Reinforcement Learning in Card Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Flyns157/SmartCardAI",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "Manager", "Management"],
    packages=setuptools.find_packages(exclude=('tests',)),
    package_data={
        'SmartCardAI': ['agents/pretrained/*',
                    ]},
    install_requires=[
        'numpy>=1.16.3',
        'termcolor'
    ],
    extras_require=extras,
    requires_python='>=3.12',
    classifiers=[
        "License :: APACHE 2.0 License",
        "Operating System :: OS Independent",
    ],
)
