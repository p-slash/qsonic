Contributing to QSOnic
======================

Discussions
-----------

We use `GitHub Discussions <https://github.com/p-slash/qsonic/discussions>`_ to organize the support and development. Feel free to post your questions and share your ideas.

Reporting Issues
----------------

Please first try reproducing the error with minimal code in a fresh terminal or notebook. Then open an `issue <https://github.com/p-slash/qsonic/issues>`_ in the GitHub repo. Provide details of the operating system and the python, numpy, numba, healpy and astropy versions you are using.

Contributing Code and Documentation
-----------------------------------

Contributions should be done via `pull requests <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_ from GitHub users' forks of the `qsonic repository <https://github.com/p-slash/qsonic>`_. You can read more about best practices `here <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests>`_. Most importantly, create small PRs and double check your work!


Install the developer version following istructions in Installation page. `dev` installation provides ``pytest, flake8, bump2version`` packages. ``pytest`` is needed for testing. You can use ``flake8`` to check for code style consistency.

**Code style** is `PEP8 <https://peps.python.org/pep-0008/>`_ with all lines limited to a maximum of 79 characters. Run ``flake8 .`` in the main repo folder before pushing your changes.

**Docstrings** are required for major functions.

**Do not** introduce additional dependencies. The combination of numpy, scipy and astropy can achieve pretty much everything you need.

**Provide** unit tests to check your functions are working as expected.

**Update** docs/, especially tutorials and installation, if needed. API will be automatically generated from docstrings.

Testing cheat sheet
-------------------

Running the following commands before pushing your branch will save you time & a lot emails from GitHub actions.

.. code:: shell

    flake8 .
    pytest
    mpirun -np 2 pytest --mpi

Creating tutorials from notebooks
---------------------------------

You can export your jupyter notebook as ReStructured Text (rst) ``File -> export as (save as) -> ReStructured Text (rst)``, and upload it to docs/source/examples. Modify this file in order to make it easy to read and provide detailed explanations between cells.
