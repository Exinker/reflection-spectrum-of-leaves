import json
import os

import numpy as np

from .alias import Array, NanoMeter, Number
from .handlers import reduce_resolution


class WavelengthCalibration:

    def __init__(self, filename: str = os.path.join('.', 'core', '.wl'), factor: int = 1) -> None:
        self.filename = filename
        self.factor = factor
        self._wavelength = None

    @property
    def wavelength(self) -> Array[float]:
        if self._wavelength is None:
            with open(os.path.join('.', 'core', 'factory_config.json'), 'r') as f:
                factory_config = json.load(f)

                direction = -1 if factory_config['reverse'] else 1
                start = factory_config['start']
                end = factory_config['end']

            filepath = os.path.join('.', self.filename)
            with open(filepath, 'r') as file:
                data = np.genfromtxt(file, delimiter='\t')

            wavelength = data[:, 0]
            wavelength = wavelength[::direction][start:end][::direction]
            wavelength = reduce_resolution(wavelength, factor=self.factor)

            self._wavelength = wavelength

        return self._wavelength

    def transform(self, value: NanoMeter) -> Number:
        """Transform wavelength to number."""
        assert min(self.wavelength) <= value <= max(self.wavelength), f'Wavelength {value:.2f} [nm] is out of the range!'

        return np.argmin(abs(self.wavelength - value))
