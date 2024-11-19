.. Predator Prey App documentation master file, created by
   sphinx-quickstart on Sun Dec 24 12:26:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Predator Prey Service's documentation!
=============================================

**Predator-Prey Service** is responsible for managing a Predator-Prey Multi-Agent RL Environment.
It allows to start the environment in two modes: *Train* mode or *Simulation* mode.

- In the first mode (*Train*), it interacts with a distributed Replay Buffer to store agents' experiences and subscribes to policy updates coming from the Learner Service.
- In *Simulation* mode, it uses the latest policy (saved from a previous training phase) to start a simulation. In this case, it accepts a seed as input to change the initial positions of the agents.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
