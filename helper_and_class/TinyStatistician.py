import numpy as np
from math import sqrt


class TinyStatistician():
    """
    A utility class for basic statistical calculations.
    All methods are static and can be used without instantiating the class.
    """

    @staticmethod
    def mean(array):
        """
        Calculates the mean of a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The mean of the values,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            print("Not the correct type")
            return None
        if len(array) == 0:
            print("Length is 0")
            return None
        result = 0.0
        for element in array:
            if not isinstance(element, (int, float)):
                print("Elements are not floats")
                return None
            result += element
        result = result / len(array)
        return float(result)

    @staticmethod
    def median(array):
        """
        Calculates the median of a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The median of the values,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        length = len(array)
        if length == 0:
            return None
        if length == 1:
            return float(array[0])
        for element in array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()
        if length % 2 != 0:
            return float(array[int(length / 2)])
        else:
            m1 = array[int(length / 2) - 1]
            m2 = array[int(length / 2)]
            return float((m1 + m2) / 2)

    @staticmethod
    def quartile(array):
        """
        Calculates the first and third quartiles of a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            list[float] | None: The first and third quartiles,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        length = len(array)
        if length == 0:
            return None
        if length == 1:
            return [float(array[0]), float(array[0])]
        for element in array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()
        quart = length / 4
        return [float(array[int(quart)]), float(array[int(quart * 3)])]

    @staticmethod
    def percentile(array, p):
        """
        Calculates the pth percentile of a list of numerical values
        using linear interpolation.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.
            p (float): Percentile to calculate (0 <= p <= 100).

        Returns:
            float | None: The value of the percentile,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        if len(array) == 0:
            return None
        if not (0 <= p <= 100):
            return None
        array = sorted(array)
        n = len(array)
        if n == 1:
            return float(array[0])
        if p == 0:
            return float(array[0])
        if p == 100:
            return float(array[-1])

        rank = (p / 100) * (n - 1)
        lower_index = int(rank)
        upper_index = lower_index + 1
        fraction = rank - lower_index

        lower_value = array[lower_index]
        upper_value = array[upper_index]
        result = lower_value + fraction * (upper_value - lower_value)
        return float(result)

    @staticmethod
    def var(array):
        """
        Calculates the variance of a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The variance of the values,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        if len(array) == 0:
            return None
        if len(array) == 1:
            return 0.0
        m = TinyStatistician.mean(array)
        if m is None:
            return None
        squared_diffs = [(x - m) ** 2 for x in array]
        sum_squared_diffs = sum(squared_diffs)
        result = sum_squared_diffs / (len(array) - 1)
        return float(result)

    @staticmethod
    def std(array):
        """
        Calculates the standard deviation of a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The standard deviation of the values,
                or None if the input is invalid.
        """
        v = TinyStatistician.var(array)
        if v is None:
            return None
        return sqrt(v)

    @staticmethod
    def min(array):
        """
        Finds the minimum value in a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The minimum value,
                or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        return float(min(array))

    @staticmethod
    def max(array):
        """
        Finds the maximum value in a list of numerical values.

        Args:
            array (list | set | tuple | np.ndarray):
                A collection of numerical values.

        Returns:
            float | None: The maximum value, or None if the input is invalid.
        """
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        return float(max(array))
