# Predator-Prey Service

The Predator-Prey Service is responsible for managing a Predator-Prey Multi-Agent RL Environment.
It allows to start the environment in two modes: *Train* mode or *Simulation* mode.

- In the first mode (*Train*), it interacts with a distributed Replay Buffer to store agents' experiences and subscribes to policy updates coming from the Learner Service.
- In *Simulation* mode, it uses the latest policy (saved from a previous training phase) to start a simulation. In this case, it accepts a seed as input to change the initial positions of the agents.

## Usage

To deploy `predator-prey-service` alongside the other microservices, follow the instructions provided in the [Bootstrap](https://github.com/DistributedMARLPredatorPrey/bootstrap) repository.

## License

Predator-Prey Service is licensed under the GNU v3.0 License. See the [LICENSE](./LICENSE) file for details.

## Author

- Luca Fabri ([w-disaster](https://github.com/w-disaster))
