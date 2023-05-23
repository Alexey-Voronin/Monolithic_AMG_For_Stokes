
class SystemIterator(object):
    """System Iterator Interface.

    Designed to generate increasingly finer rediscretization of a desired
    problem.
    """

    count      = None
    build_time = -1 # time to construct the system

    def __init__(self, system_params, max_dofs):
        """Initialize System Iterator."""
        self.system_params = system_params
        self.max_dofs      = max_dofs

    def __repr__(self):
        """Return string representation of SystemIterator."""
        pass

    def __iter__(self):
        self.count =  0
        return self

    def __next__(self):
        """Return next system in the sequence."""
        pass

    def get_build_time(self):
        return self.build_time
