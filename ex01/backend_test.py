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

        def evaluate(A, b, reference, delta = 10):
            nonlocal comments
            try:
                x = self.module.solveLinearSystem(A, b)
                
                eps_student = np.linalg.norm(np.dot(A, x) - b)
                eps_reference= np.linalg.norm(np.dot(A, reference) - b)
            
                if eps_student < eps_reference * delta or eps_student < 1e-12:
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Identity
        comments += "Identity case "

        A = np.identity(30)
        b = np.ones(30)
        for i in range(30):
            b[i] = (i + 1)
        reference = np.copy(b)
        evaluate(A, b, reference)

        # 30x30 floats
        comments += "30x30 case "

        A = np.tril(np.ones((30, 30)))
        tmp = np.triu(np.ones((30, 30)))
        for i in range(30):
            A[i] *= i + 1
            A[:, i] /= i + 1
            tmp[i] *= (i + 1) ** 2
            tmp[:, i] /= (i + 1) ** 2
        A = A.dot(tmp)
        A *= np.pi / np.e
        b = np.ones(30)
        
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference)

        # 30x30 instable
        comments += "30x30 unstable case "

        A = np.tril(np.ones((30, 30)))
        tmp = np.triu(np.ones((30, 30)))
        for i in range(30):
            A[i] *= i + 1
            A[:, i] /= i + 1
            tmp[i] *= (i + 1) ** 2
            tmp[:, i] /= (i + 1) ** 2
        A = A.dot(tmp)
        A *= np.pi / np.e
        for i in range(30):
            A[i] *= np.exp(i)
            A[:, i] /= np.exp(i)
        b = np.ones(30)
        
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference)

        # 10x10 Pivoting
        comments += "10x10 Pivoting case "

        A = np.triu(np.ones((10, 10)))
        A = np.roll(A, 1, axis=0)
        b = np.ones(10)
        
        reference = np.linalg.solve(A, b)
        evaluate(A, b, reference)
        
        #optional case for infintely many solutions
        
        comments += " optional 10x10 case with infinite solutions "
        
        A = np.identity(10)
        A[3,3] = 0
        b = np.ones(10)
        b[3] = 0
        
        reference = np.ones(10)
        evaluate(A, b, reference)
        

        
        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task b
    #############################################

    def testB(self, l: list, task):
        comments = ""

        def evaluate(A, b, reference):
            nonlocal comments
            try:
                if (self.module.isConsistent(np.copy(A), np.copy(b)) == reference):
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "


        def verifyConsitency(A,b):
            return np.linalg.matrix_rank(A) == np.linalg.matrix_rank(np.c_[A,b])
        
        # 10x10 upper triangular
        comments += "10x10 upper triangle case "

        A = np.triu(np.ones((10, 10)))
        b = np.ones(10)
        
        reference = verifyConsitency(A,b)
        evaluate(A, b, reference)

        # 10x10 floats
        comments += "10x10 case with floating numbers "

        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.
        A[-1] = A[0] + A[-2]
        b = np.ones(10)
        # Solve the linear system of equations
        
        reference = verifyConsitency(A,b)
        evaluate(A, b, reference)
        
        # 10x10 infinte solutions
        comments += "optional 10x10 case with infinite solutions "
        A = np.identity(10)
        A[3,3] = 0
        b = np.ones(10)
        b[3] = 0
        
        reference = verifyConsitency(A,b)
        evaluate(A, b, reference)


        result = [task, comments]
        print(result)
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

        result = self.performTest(self.testA, '1.1a)')
        evaluateResult("1a)", result)

        result = self.performTest(self.testB, '1.1b)')
        evaluateResult("1b)", result)

        return l


tester = Tester()
overall_result = []
tester.runTests(backend, overall_result)

