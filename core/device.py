import os
import time
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum, auto
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

from pyspectrum.device_factory import UsbID
from pyspectrum.spectrometer import Spectrometer, FactoryConfig

from .alias import Array, Hz, MilliSecond, Number
from .config import FILEDIR
from .wavelength import WavelengthCalibration


class DeviceConfig:

    def __init__(self, omega: Hz, tau: MilliSecond = 2) -> None:
        assert isinstance(omega, (int, float)), 'Частота регистрации `omega` должно быть числом!'
        assert 0.1 <= omega <= 100, 'Частота регистрации `omega` должна лежать в диапазоне [0.1; 100] Гц!'
        assert isinstance(tau, (int, float)), 'Базовое время экспозиции `tau` должно быть числом!'
        assert 1 <= tau <= 1_000, 'Базовое время экспозиции `tau` должно лежать в диапазоне [2; 1_000] мс!'

        self._omega = omega
        self._tau = tau
        self._buffer_size = self.calculate_buffer_size(omega=omega, tau=tau)
        self._factor = 1

    @property
    def omega(self) -> float:
        """Частота регистрации (Гц)."""
        return self._omega

    @property
    def tau(self) -> MilliSecond:
        """Базовое время экспозиции (мкс)."""
        return self._tau

    @property
    def buffer_size(self) -> int:
        """Количество накоплений во времени."""
        return self._buffer_size

    @property
    def factor(self) -> Number:
        """Количество накоплений в пространстве."""
        return self._factor

    @staticmethod
    def calculate_buffer_size(omega: Hz, tau: MilliSecond) -> int:
        """Рассчитать размер буфера."""
        assert Decimal(1e+3) / Decimal(omega) % Decimal(tau) == 0, 'Частота регистрации `omega` должна быть кратна базовому времени экспозиции `tau`!'

        return int(Decimal(1e+3) / Decimal(omega) / Decimal(tau))

    @property
    def scale(self) -> float:
        """Коэффициент интенсивности."""
        return 100 / (2**16 - 1)


class Device(ABC):

    def __init__(self, config: DeviceConfig) -> None:

        self._config = config
        self._device = Spectrometer(
            UsbID(),
            factory_config=FactoryConfig.load(os.path.join('.', 'core', 'factory_config.json'))
        )

        self._wavelength = None
        self._dark_data = None

    @property
    def config(self) -> DeviceConfig:
        return self._config

    # --------        read        --------
    def read(self, n_frames: int, *args, **kwargs):
        """Начать чтение `n_frames` кадров."""
        config = self.config

        # n_frames
        if n_frames == -1:  # while cycle
            n_frames = 2**24 - 1

        n_frames = (n_frames//config.buffer_size) * config.buffer_size
        assert n_frames < 2**24, 'Общее количество накполений `n_frames` должно быть менее 2**24!'

        # setup
        self._device.set_config(
            exposure=self.config.tau,
            n_times=config.buffer_size,
        )

        # read
        self._read(n_frames, *args, **kwargs)

    # --------        dark_data        --------
    @property
    def dark_data(self) -> Array[float]:
        """Темновой сигнал."""
        assert self._dark_data is not None, 'Calibrate device to dark data!'

        return self._dark_data

    def calibrate_dark_signal(self, n_frames: int, filename: str = 'dark.data', show: bool = False) -> None:
        """Калибровать устройство на темновой сигнал по `n_frames` накоплений."""

        self._device.set_config(
            exposure=self.config.tau,
            n_times=n_frames,
        )

        data = np.mean(self._device.read_raw().intensity, axis=0)
        data *= self.config.scale
        data[data==100] = np.nan

        # show
        if show:
            figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

            x = np.arange(1, len(data)+1)
            y = data
            plt.plot(
                x, y,
                color='black', linestyle='-',
            )

            plt.xlabel('number')
            plt.ylabel('$I_{d}$, %')

            plt.grid(color='grey', linestyle=':')

            plt.show()

        # # save
        # filepath = os.path.join('.', 'data', filename)
        # with open(filepath, 'w') as file:
        #     np.savetxt(file, data)

        #
        self._dark_data = data

    # --------        wavelength        --------
    @property
    def wavelength(self) -> Array[float]:

        if self._wavelength is None:
            self._wavelength = WavelengthCalibration(
                factor=self.config.factor,
            ).wavelength

        return self._wavelength

    # --------        handlers        --------
    @abstractmethod
    def _show(self, data: Array[float], xlim: tuple[float, float] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def _read(self, n_frames: int, show: bool = False, xlim: tuple[float, float] | None = None):
        raise NotImplementedError


class EmissionDevice(Device):

    def __init__(self, config: DeviceConfig) -> None:
        super().__init__(config)

    # --------        handlers        --------
    def _read(self, n_frames: int, show: bool = False, xlim: tuple[float, float] | None = None, save: bool = False):
        """Начать чтение `n_frames` кадров."""
        config = self.config

        # read
        for i in range(n_frames//config.buffer_size):
            completed = 100 * (i + 1) / (n_frames//config.buffer_size)

            data = np.mean(self._device.read_raw().intensity, axis=0)
            data *= self.config.scale
            data[data==100] = np.nan
            data -= self.dark_data

            #
            if show:
                display.clear_output(wait=True)
                self._show(data=data, completed=completed, xlim=xlim)

        # save
        if save:
            filedir = os.path.join('.', 'data', FILEDIR)
            if not os.path.isdir(filedir):
                os.mkdir(filedir)

            filepath = os.path.join(filedir, f'{int(time.time())}.csv')
            pd.DataFrame({
                'wavelength': self.wavelength,
                'intensity': data,
            }).to_csv(
                filepath,
                index=False,
            )

    def _show(self, data: Array[float], completed: float, xlim: tuple[float, float] | None = None) -> None:

        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

        x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
        y = data
        plt.plot(
            x, y,
            color='black', linestyle='-',
        )

        content = [
            fr'$\omega$: {self.config.omega} [Hz]',
            fr'$\delta{{t}}$: {self.config.buffer_size}',
        ]
        plt.text(
            .95, .95,
            '\n'.join(content),
            fontsize=12,
            ha='right', va='top',
            transform=plt.gca().transAxes,
        )

        content = [
            fr'{completed:>3.0f}%',
        ]
        plt.text(
            .05, .05,
            '\n'.join(content),
            fontsize=12,
            ha='left', va='bottom',
            transform=plt.gca().transAxes,
        )

        plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
        plt.ylabel('$I$, %')

        if xlim: plt.xlim(xlim)

        plt.grid(color='grey', linestyle=':')
        plt.pause(.001)



class Units(Enum):
    A = auto()
    T = auto()


class AbsorptionDevice(Device):

    def __init__(self, config: DeviceConfig) -> None:
        super().__init__(config)

        self._base_data = None

    # --------        base_data        --------
    @property
    def base_data(self) -> Array[float]:
        """Сигнал источника излучения."""
        assert self._base_data is not None, 'Calibrate device to base data!'

        return self._base_data

    def calibrate_base_signal(self, n_frames: int, filename: str = 'base.data', show: bool = False) -> None:
        """Калибровать устройство на сигнал источника излучения по `n_frames` накоплений."""

        self._device.set_config(
            exposure=self.config.tau,
            n_times=n_frames,
        )

        data = np.mean(self._device.read_raw().intensity, axis=0)
        data *= self.config.scale
        data[data==100] = np.nan

        data -= self.dark_data

        # show
        if show:
            figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

            x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
            y = data
            plt.plot(
                x, y,
                color='grey', linestyle='-',
            )

            plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
            plt.ylabel('$I_{0}$, %')

            plt.grid(color='grey', linestyle=':')

            plt.show()

        # save
        filepath = os.path.join('.', 'data', filename)
        with open(filepath, 'w') as file:
            np.savetxt(file, data)

        #
        self._base_data = data

    # --------        read        --------
    def _read(self, n_frames: int, show: bool = False, save: bool = False, filename: str = 'data.data', xlim: tuple[float, float] | None = None, units: Units = Units.A):
        """Начать чтение `n_frames` кадров."""
        config = self.config

        # n_frames
        n_frames = (n_frames//config.buffer_size) * config.buffer_size  # FIXME: 
        assert n_frames < 2**24, 'Общее количество накполений `n_frames` должно быть менее 2**24!'

        # setup
        self._device.set_config(
            exposure=self.config.tau,
            n_times=config.buffer_size,
        )

        for i in range(n_frames//config.buffer_size):
            completed = 100 * (i + 1) / (n_frames//config.buffer_size)

            # read
            data = np.mean(self._device.read_raw().intensity, axis=0)
            data *= self.config.scale
            data[data==100] = np.nan
            data -= self.dark_data

            # show
            if show:
                display.clear_output(wait=True)
                self._show(data=data, completed=completed, xlim=xlim, units=units)

        # save
        filepath = os.path.join('.', 'data', 'absorbance.csv')
        pd.DataFrame(
            {
                'wavelength': self.wavelength,
                'absorbance': np.log10(self.base_data / data)
            }
        ).to_csv(
            filepath,
        )

    def _show(self, data: Array[float], completed: float, xlim: tuple[float, float] | None = None, units: Units = Units.A) -> None:

        figure, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

        # 
        plt.sca(ax_left)

        x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
        y = self.base_data
        plt.plot(
            x, y,
            color='grey', linestyle='-',
            label='$I_{0}$'
        )

        x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
        y = data
        plt.plot(
            x, y,
            color='black', linestyle='-',
            label='$I$'
        )

        content = [
            fr'$\omega$: {self.config.omega} [Hz]',
            fr'$\delta{{t}}$: {self.config.buffer_size}',
        ]
        plt.text(
            .95, .95,
            '\n'.join(content),
            fontsize=12,
            ha='right', va='top',
            transform=plt.gca().transAxes,
        )

        content = [
            fr'{completed:>3.0f}%',
        ]
        plt.text(
            .05, .05,
            '\n'.join(content),
            fontsize=12,
            ha='left', va='bottom',
            transform=plt.gca().transAxes,
        )

        plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
        plt.ylabel('$I$, %')

        if xlim: plt.xlim(xlim)

        plt.grid(color='grey', linestyle=':')
        plt.legend(loc='upper left')

        # 
        plt.sca(ax_right)

        x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
        y = np.log10(self.base_data / data) if units==Units.A else data / self.base_data
        plt.plot(
            x, y,
            color='black', linestyle='-',
        )

        plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
        plt.ylabel('$A$' if units==Units.A else '$T$')

        if xlim: plt.xlim(xlim)

        plt.grid(color='grey', linestyle=':')

        #
        plt.pause(.001)
