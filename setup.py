from distutils.core import setup
setup(
  name = 'chesslab',         # How you named your package folder (MyLib)
  packages = ['chesslab'],   # Chose the same as "name"
  version = '0.5',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library for developing evaluation functions using Neural Networks',   # Give a short description about your library
  author = 'Hector Juarez',                   # Type in your name
  author_email = 'hjuarezl1400@alumno.ipn.mx',      # Type in your E-Mail
  url = 'https://github.com/yniad/chesslab',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/yniad/chesslab/archive/refs/tags/0.5.tar.gz',    # I explain this later on
  keywords = ['chess', 'chesslab', 'evaluation function','agent'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'chess',
          'numpy',
          'torch'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
  ],
)
