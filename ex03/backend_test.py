import numpy as np
import sys
import traceback
import backend


class Tester:
    def __init__(self):
        self.module = None
        self.runtime = 300

    #############################################
    # Task a
    #############################################

    def testA(self, l: list, task):
        comments = ""

        def evaluate(sourceBase, targetBase, reference):
            nonlocal comments
            try:
                A = self.module.changeBase(sourceBase, targetBase)
                if ((np.abs(A - reference) < 1e-6).all()):
                    comments += "passed. "
                else:
                    if ((np.abs(np.linalg.inv(A) - reference) < 1e-6).all()):
                        comments += "inverse. "
                    else:
                        comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # 45 degree case
        comments += "45 degree case "
        base = np.identity(3).T
        sourceBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                      np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]),
                      np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)])]
        targetBase = list(base)
        reference = np.array(sourceBase).T

        evaluate(sourceBase, targetBase, reference)

        # 45 degree to -45 degree case
        comments += "45 degree to -45 degree case "
        sourceBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                      np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]),
                      np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)])]
        targetBase = [np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)]),
                      np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                      np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)])]
        reference = np.linalg.inv(np.array(targetBase).T).dot(np.array(sourceBase).T)

        evaluate(sourceBase, targetBase, reference)

        # Roll case
        comments += "Roll case "

        base = np.identity(10) + np.roll(np.identity(10), 2, axis=0)
        base[-1, 0] = 0
        sourceBase = list(base)
        base = np.identity(10) + np.roll(np.identity(10), 1, axis=0) + np.roll(np.identity(10), -1, axis=0)
        base[0, -1] = 0
        base[-1, 0] = 0
        targetBase = list(base)
        reference = np.linalg.inv(np.array(targetBase).T).dot(np.array(sourceBase).T)
        evaluate(sourceBase, targetBase, reference)

        result = [task, comments]
        print('\n', result)
        l.extend(result)

    #############################################
    # Task b
    #############################################

    def testB(self, l: list, task):
        comments = ""

        def evaluate(base, subBase, reference):
            nonlocal comments
            try:
                isSubSpace = self.module.spansSubSpace(base, subBase)
                if (isSubSpace == reference):
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # R2-R2 case
        comments += "R2-R2 case "

        base = list(np.identity(2))
        subBase = list(np.identity(2))
        evaluate(base, subBase, True)

        # R3-R2 case°
        comments += "R3-R2 case "

        base = list(np.identity(3))
        subBase = list(np.identity(3))[:-1]
        evaluate(base, subBase, True)

        # XY-XZ case
        comments += "XY-XZ case "

        base = list(np.identity(3))[:-1]
        subBase = list(np.identity(3))[0::2]
        evaluate(base, subBase, False)

        # R2-R3 case
        comments += "R2-R3 case "

        base = list(np.identity(3))[:-1]
        subBase = list(np.identity(3))
        evaluate(base, subBase, False)

        # R1-Zero case
        comments += "R1-Zero case "

        base = [np.array([1.])]
        subBase = [np.array([0.])]
        evaluate(base, subBase, True)

        result = [task, comments]
        print('\n', result)
        l.extend(result)

    def performTest(self, func, task):
        l = []
        try:
            func(l, task)
            return l
        except Exception as e:
            return []

    def runTests(self, module, l):
        self.module = module

        def evaluateResult(task, result):
            if (len(result) == 0):
                l.append([task, 0, "Interrupt."])
            else:
                l.append(result)

        result = self.performTest(self.testA, "3.1a)")
        evaluateResult("3.1a)", result)

        result = self.performTest(self.testB, "3.1b)")
        evaluateResult("3.1b)", result)

        return l


tester = Tester()
overall_result = []
tester.runTests(backend, overall_result)