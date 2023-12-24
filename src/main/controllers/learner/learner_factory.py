from src.main.controllers.learner.learner import Learner


class LearnerFactory:
    @staticmethod
    def create_learners(buffers, par_services, num_states, num_actions):
        """
        Create a new Learner for each buffer passed as parameter
        :param buffers: Buffers
        :param par_services: ParameterServices
        :param num_states: number of states
        :param num_actions: number of actions
        :return:
        """
        return [
            Learner(
                buffers[i],
                par_services[i],
                num_states,
                num_actions,
                len(par_services[i]),
            )
            for i in range(len(buffers))
        ]
