class Worker(object):
    def __init__(self, args):
        self.args = args

        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.accs = 0
        self.model = None
        self.orders = None
        self.dones = None
        self.cvs = None
        self.steps = None
        self.features = None
        self.ff = None
        self.steps = None
