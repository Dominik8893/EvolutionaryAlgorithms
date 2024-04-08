import sys
import sys
import math as math
import matplotlib.pyplot as plt
import numpy as np


class SimulatedAnnealing:
    def __init__(self, file_path, MaxIterations=9999999, mode="exponential", temperature_initial=1e6, alpha=0.9999, beta=0.0001, maximum_error=1e-6):
        self.temperature_initial = temperature_initial
        self.maximum_error = maximum_error
        self.MaxIterations = MaxIterations
        self.mode = mode
        if self.mode == "exponential":
            self.cooling_factor = alpha
        elif self.mode == "inverse":
            self.cooling_factor = beta
        else:
            print("Wrong mode provided, quitting.")
            sys.exit()

        with open(file_path, "r") as CheckLengthFile:
            for MeasurementNumber, line in enumerate(CheckLengthFile):
                pass
        self.MeasurementNumber = MeasurementNumber + 1
        measurements_array = np.zeros((self.MeasurementNumber, 2))

        with open(file_path, "r") as model_data:
            for i, line in enumerate(model_data.readlines()):
                measurements_array[i] = [float(value) for i, value in enumerate(line.split())]  # input and output values

        self.MeasurementsArray = measurements_array
        self.RealOutputMean = np.mean(self.MeasurementsArray)
        self.Coefficients = np.random.uniform(-10, 10, 3) #a b and c
        self.CoefficientsTemperatures = np.full(3, self.temperature_initial)
        self.EstimatedOutputValues = self.MainLoop()
        self.visualisation()

    def MainLoop(self):

        Iterator = 1
        mse_current = self.EstimatedOutput(self.Coefficients, MSE=True)

        while True:
            # print(self.function_coefficients_temperatures)
            NewCoefficientsTemperatures = np.random.normal(0, self.CoefficientsTemperatures, 3)
            NewCoefficients = self.Coefficients + NewCoefficientsTemperatures
            mse_new = self.EstimatedOutput(NewCoefficients, MSE=True)

            if mse_current > mse_new:
                # self.function_coefficients = new_function_coefficients
                self.Coefficients = NewCoefficients
            else:
                for i in range(3):
                    random_value = np.random.uniform()  # value from 0 to 1
                    if random_value < math.exp((mse_current - mse_new) / self.CoefficientsTemperatures[i]):
                        self.Coefficients[i] = NewCoefficients[i]


            if abs(mse_new - mse_current) < self.maximum_error or Iterator>self.MaxIterations:  # mse --- mean squared error
                results = self.EstimatedOutput(self.Coefficients, False)
                print(Iterator)
                return results
            else:
                self.CoefficientsTemperatures = self.cooling()
                mse_current = self.EstimatedOutput(self.Coefficients, MSE=True)
                Iterator += 1


    def EstimatedOutput(self, Coefficients, MSE=False): # for mean squared error, pass the True
        EstimatedValues = np.zeros(self.MeasurementNumber)

        for i in range(len(EstimatedValues)):
            EstimatedValues[i] = Coefficients[0] * (self.MeasurementsArray[i][0] ** 2 - Coefficients[1] * math.cos(Coefficients[2] * math.pi * self.MeasurementsArray[i][0]))

        if MSE:
            MSE = np.mean((self.MeasurementsArray[:, 1] - EstimatedValues) ** 2)
            return MSE
        else:
            return EstimatedValues


    def cooling(self):

        if self.mode == "exponential":
            cooling_function = self.cooling_factor * self.CoefficientsTemperatures
        elif self.mode == "inverse":
            cooling_function = self.CoefficientsTemperatures / (1 + (self.cooling_factor * self.CoefficientsTemperatures))

        return cooling_function


    def visualisation(self):
        plt.scatter(self.MeasurementsArray[:, 0], self.EstimatedOutputValues)
        plt.scatter(self.MeasurementsArray[:, 0], self.MeasurementsArray[:, 1])
        plt.legend(["Estimated", "Real"])
        plt.show()

if __name__ == "__main__":
    result = SimulatedAnnealing(r".venv/Scripts/model1.txt",30000)