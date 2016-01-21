from distutils.core import setup

setup(
        name='dataViz',
        version='1.0.0',
        packages=['dataviz', 'dataviz.tests', 'dataviz.statistics'],
        url='',
        license='Apache 2.0',
        author='Socrats',
        author_email='',
        description='Data vizualization scripts and tools for python', requires=['matplotlib', 'pandas', 'sympy']
)
