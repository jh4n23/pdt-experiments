import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent
""" Given a model, a test set and an epsilon value, run APGD attack and print accuracy """

class CsecTester():

    def __init__(
        self,
        model,
    ):
        self.model = model

    def attack(self, x_test, epsilon):
        attack = AutoProjectedGradientDescent(estimator=self.model, eps=epsilon)
        adv_test = attack.generate(x=x_test)
        return self.model.predict(adv_test)

    @staticmethod
    def eval(preds, y_test):
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        return acc

    def run(self, x_test, y_test, epsilon):
        preds = self.attack(x_test, epsilon)
        return self.eval(preds, y_test)