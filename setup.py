from setuptools import setup, find_packages

REQUIRED = ['gym', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'flask']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rl-visualization',
    version='0.1',
    packages=['rl_visualization',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    long_description=long_description,
    url='https://github.com/LucasAlegre/rl-visualization',
    license="MIT",
    description='Reinforcement Learning Visualization.'
)