"""This module contains classes for checking and detecting events in a trial."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

## to remove
import matplotlib.pyplot as plt

import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.utils.mocap as mocap

FOOT_STRIKE = "Foot Strike"
FOOT_OFF = "Foot Off"
LEFT = "Left"
RIGHT = "Right"
SIDES = [LEFT, RIGHT]
EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]


class _BaseEventChecker(ABC):
    """Abstract class for event checkers.

    This class provides a common interface for checking events in a trial,
    which makes them interchangeable.
    """

    @abstractmethod
    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the events in the trial.

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the events are correct, False otherwise.
            list | None: A list of incorrect time slices,
                or None if the events are correct.
        """
        raise NotImplementedError


class SequenceEventChecker(_BaseEventChecker):
    """A class for checking the sequence of events in a trial.

    This class provides a method to check the sequence of events in a trial.
    It checks the sequence of event labels and contexts.
    """

    _TIME_COLUMN = io._EventInputFileReader.COLUMN_TIME
    _LABEL_COLUMN = io._EventInputFileReader.COLUMN_LABEL
    _CONTEXT_COLUMN = io._EventInputFileReader.COLUMN_CONTEXT
    _SEQUENCE = pd.DataFrame(
        {
            "current": [
                "Foot Strike Right",
                "Foot Off Left",
                "Foot Strike Left",
                "Foot Off Right",
            ],
            "next": [
                "Foot Off Left",
                "Foot Strike Left",
                "Foot Off Right",
                "Foot Strike Right",
            ],
            "previous": [
                "Foot Off Right",
                "Foot Strike Right",
                "Foot Off Left",
                "Foot Strike Left",
            ],
        }
    ).set_index("current")

    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the sequence of events in the trial.

        In a normal gait cycle, the sequence of events is as follows:
        1. Foot Strike (right)
        2. Foot Off (left)
        3. Foot Strike (left)
        4. Foot Off (right)

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the sequence is correct, False otherwise.
            list | None: A list time slice of incorrect sequence,
            or None if the sequence is correct.

        """
        if events is None:
            raise ValueError("Trial does not have events.")

        incorrect_times = []

        incorrect_labels = self._check_labels(events)
        incorrect_contexts = self._check_contexts(events)

        if incorrect_labels:
            incorrect_times.append(incorrect_labels)
        if incorrect_contexts:
            incorrect_times.append(incorrect_contexts)

        return not bool(incorrect_times), incorrect_times if incorrect_times else None

    def _check_labels(self, events: pd.DataFrame) -> list[tuple]:
        """Check alternating sequence of event labels.

        Expected sequence of event labels:
        1. Foot Strike
        2. Foot Off
        3. Foot Strike
        4. Foot Off

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        last_label = None
        last_time = None
        for i, label in enumerate(events[self._LABEL_COLUMN]):
            time = events[self._TIME_COLUMN].iloc[i]
            if label == last_label:
                incorrect_times.append((last_time, time))

            last_time = time
            last_label = label
        return incorrect_times

    def _check_contexts(self, events: pd.DataFrame) -> list[tuple]:
        """Check sequence of contexts of events.

        Expected sequence of event contexts:
        1. Right
        2. Right
        3. Left
        4. Left

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        # Check the occurrence of the context in windows of 3 events.
        for i in range(len(events) - 3):
            max_occurance = (
                events[self._CONTEXT_COLUMN].iloc[i : i + 3].value_counts().max()
            )

            # If the context occurs more than twice in the window, it is incorrect.
            if max_occurance > 2:
                incorrect_times.append(
                    (
                        events[self._TIME_COLUMN].iloc[i],
                        events[self._TIME_COLUMN].iloc[i + 3],
                    )
                )

        return incorrect_times


class BaseEventDetection(ABC):
    _TIME_COLUMN = io._EventInputFileReader.COLUMN_TIME
    _LABEL_COLUMN = io._EventInputFileReader.COLUMN_LABEL
    _CONTEXT_COLUMN = io._EventInputFileReader.COLUMN_CONTEXT
    _ICON_COLUMN = io._EventInputFileReader.COLUMN_ICON

    """Abstract class for event detectors.

    This class provides a common interface for detecting events of a specific type (Foot Strike vs Foot Off) for one side in a trial,
    which makes them interchangeable.
    """

    def __init__(
        self,
        configs: mapping.MappingConfigs,
        context: str,
        label: str,
        offset: float = 0.0,
    ):
        """Initializes a new instance of the BaseEventDetection class for an event type on a single side.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.

        """
        self._configs = configs
        self._context = context
        self._label = label
        self._offset = offset

    def _create_data_frame(self, times: np.ndarray) -> pd.DataFrame:
        """Creates a DataFrame from the detected events.

        Args:
            times: The detected event times.
            context: The context of the detected events.
            label: The label of the detected events.

        Returns:
            pd.DataFrame: A DataFrame containing the detected events.
        """
        contexts = [self._context] * len(times)
        labels = [self._label] * len(times)
        icons = [1 if self._label == FOOT_STRIKE else 2] * len(times)

        table = {
            self._TIME_COLUMN: times,
            self._LABEL_COLUMN: labels,
            self._CONTEXT_COLUMN: contexts,
            self._ICON_COLUMN: icons,
        }
        # print(table)
        events = pd.DataFrame.from_dict(table)
        return events

    def detect_events(self, trial: model.Trial) -> np.ndarray:
        """Detects the events in the trial and adds an offset if there is any

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the timings of the specific event in seconds
        """
        events = self._detect_events(trial)
        return self._add_offset(events, self._offset)

    def _add_offset(self, events: np.ndarray, offset: float) -> np.ndarray:
        return events - offset

    @abstractmethod
    def _detect_events(self, trial: model.Trial) -> np.ndarray:
        """Detects the events in the trial.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the detected events.
        """
        raise NotImplementedError

    def _rotate_markers(self, points: list[xr.DataArray], trial) -> list[xr.DataArray]:
        """Rotates the 3D coordinates of the marker space, such that the progression axis is the x axis

        Args:
            points: list of xr.DataArrays containing all the marker trajectories to be rotated
            trial: The trial for which to detect the events.

        Returns:
            list[xr.DataArrays] : the same list of markers, into the rotated coordinate system
        """
        if not points:
            raise ValueError("points [list[xr.DataArray]] cannot be empty")
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        l_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.L_ANT_HIP
        )
        r_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.R_ANT_HIP
        )
        ant_hip = (l_ant_hip + r_ant_hip) / 2
        progress_axis = ant_hip - sacrum
        x_axis = xr.DataArray(
            [1, 0, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]}
        )
        angles = self._calculate_angle(progress_axis, x_axis)
        ant_hip = self._rotate_point(sacrum, sacrum, angles)
        scale = self._get_flip_scale(ant_hip - sacrum)
        for i in range(len(points)):
            point = self._rotate_point(points[i], sacrum, angles)
            points[i] = (point.T * scale).T
        return points

    @staticmethod
    def _calculate_angle(progress: xr.DataArray, axis: xr.DataArray) -> float:
        """Calculate the angle between two vectors.

        Args:
            progress: The first vector.
            axis: The second vector.

        Returns:
            float: The angle between the two vectors.
        """
        progress = progress.drop_sel(axis="z")
        axis = axis.drop_sel(axis="z")
        theta = np.arccos(
            progress.dot(axis, dim="axis")
            / (progress.meca.norm(dim="axis") * axis.meca.norm(dim="axis"))
        )
        return theta.values

    @staticmethod
    def _rotate_point(
        point: xr.DataArray, fix_point: xr.DataArray, angle: float
    ) -> xr.DataArray:
        """Rotate a point around a fixed point.

        Args:
            point: The point to rotate.
            fix_point: The fixed point.
            angle: The angle to rotate by.

        Returns:
            xr.DataArray: The rotated point.
        """
        fix = fix_point.drop_sel(axis="z")
        rel_point = point.drop_sel(axis="z")
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rel_fix = fix.to_numpy()
        np_rel_point = rel_point.to_numpy()
        two_d_point = np.empty(np_rel_point.shape)
        for i in range(np_rel_point.shape[1]):
            two_d_point[:, i] = np_rel_point[:, i] - rel_fix[:, i]
            two_d_point[:, i] = two_d_point[:, i] @ rot[:, :, i].T
            two_d_point[:, i] = two_d_point[:, i] + rel_fix[:, i]
        point = point.copy(deep=True)
        point.loc["x"] = two_d_point[0]
        point.loc["y"] = two_d_point[1]
        return point

    @staticmethod
    def _get_flip_scale(rot_progress: xr.DataArray) -> list:
        if (rot_progress.loc["x"] < 0).sum() > rot_progress.loc["x"].shape[0] / 2:
            return [-1, 1, 1]
        else:
            return [1, 1, 1]


# TODO: rename GRF classes
class Grf1EventDetection(BaseEventDetection):
    def __init__(self, configs, context, label):
        super().__init__(configs, context, label)
        self.frate = 100  # TODO Hz --> take it from c3d file. How?

    def _get_sliding_window(self, time, signal, width, start):
        if self._label == FOOT_OFF:
            return signal[(time <= start) & (time > start - width)]
        else:
            return signal[(time >= start) & (time < start + width)]

    def _get_condition(self, time, start, width):
        if self._label == FOOT_OFF:
            return start - width >= time[0]
        else:
            return start + width <= time[-1]

    def _get_start_point(self, time):
        return time[0] if self._label == FOOT_STRIKE else time[-1]

    def _get_end_point(self, start, width):
        return start + width if self._label == FOOT_STRIKE else start - width

    def _move_sliding_window(self, start):
        return (
            start + (1 / self.frate)
            if self._label == FOOT_STRIKE
            else start - (1 / self.frate)
        )

    def _detect_events(self, trial):
        # based on https://doi.org/10.1016/j.gaitpost.2006.09.077
        print(f"\n\t {self._label} {self._context}\n")
        markers = trial.get_data(model.DataCategory.MARKERS)
        GRF_3d = (
            markers.sel(channel="LNormalisedGRF")
            if self._context == LEFT
            else markers.sel(channel="RNormalisedGRF")
        )
        GRF = GRF_3d.loc["z"]
        times = []
        GRF_ = (
            np.nan_to_num(GRF) if np.sum(np.isnan(GRF)) > 0 else np.copy(GRF)
        )  # check for Nans
        # b, a = sp.signal.butter(4, 20, fs=self.frate)
        # GRF_filt = sp.signal.filtfilt(b, a, GRF_)
        GRF_filt = GRF_
        time_ = GRF.time.data
        sd = np.std(GRF_filt[time_ <= time_[0] + 0.1])
        width = 0.04  # 40 ms
        start = self._get_start_point(time_)
        window = GRF_filt[time_ < self._get_end_point(start, width)]
        prev_mean = np.mean(window)
        while self._get_condition(time_, start, width):
            start = self._move_sliding_window(start)
            print(start)
            window = self._get_sliding_window(time_, GRF_filt, width, start)
            threshold = prev_mean + 3 * sd
            print("\t", threshold)
            print("\t", window)
            if np.sum(window <= threshold) == 0:
                times.append(start)
            prev_mean = np.mean(window)
        events = np.array(times)
        return events


class BaseOptimisedEventDetection(BaseEventDetection, ABC):
    """Abstract class for event detectors used with a reference for optimisation.

    This class provides a common interface for detecting events in a trial with a reference,
    which makes them interchangeable.
    """

    def __init__(
        self,
        configs: mapping.MappingConfigs,
        context: str,
        label: str,
        offset: float = 0,
        trial_ref: Union[None, model.Trial] = None,
    ):
        """Initializes a new instance of the BaseOptimisedEventDetection class for an event type on a single side.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            trial_ref: Trial to be used as reference, if any is required. Otherwise None
        """
        super().__init__(configs, context, label, offset)
        self.trial_ref = trial_ref
        self.ref_events = (
            self.get_event_times(trial_ref.events) if trial_ref is not None else None
        )
        self.min_dist = self.get_min_dist()
        self.frate = 100  # TODO Hz --> take it from c3d file. How?

    def get_event_times(self, events_df: pd.DataFrame) -> np.ndarray:
        """Gets the relevant events for this instance form the complete events table of the trial

        Args:
            events_df: complete event table of a trial

        Returns:
            np.ndarray: array containing the event timings for the event and side of the instance
        """
        events = events_df.loc[
            (events_df["context"] == self._context)
            & (events_df["label"] == self._label),
            "time",
        ]
        return events.to_numpy()

    def get_min_dist(self):
        """
        Gets the minimum distance in terms of time duration between two reference events of this instance
        """
        if self.ref_events is not None:
            return np.min(self.ref_events[1:] - self.ref_events[:-1])
        else:
            return 0.7  # TODO: very arbitrary value

    def _get_accuracy(self, times: np.ndarray):
        """Computes the accuracy of detected events with the true reference events

        Args:
            times: array containing the detected events timings

        Returns:
            np.ndarray: array containing the errors of the detected events with the reference events
            float [0; 1]: fraction of events that have not been detected (missed events)
            float [0; 1]: fraction of events detected that do not take place in reality (excess events)
        """
        events_ref = self.ref_events
        rad_ = 0.5 * self.min_dist
        rad = 0.2
        in_ = np.array([])
        out_ = np.copy(times)
        missed = 0
        diff_list = np.array([])
        if events_ref is None:
            raise ValueError("Reference trial must be provided")
        for ev_ref in events_ref:
            tmp = out_[(out_ <= ev_ref + rad_) & (out_ >= ev_ref - rad_)]
            if not len(tmp):
                missed += 1
            else:
                if len(tmp) == 1:
                    ev = tmp[0]
                else:
                    ev = tmp[np.argmin(np.abs(tmp - ev_ref))]
                out_ = out_[out_ != ev]
                in_ = np.append(in_, ev)
                diff_list = np.append(diff_list, ev_ref - ev)
        offset = self._compute_offset(
            np.mean(diff_list), self._compute_quantiles(diff_list)
        )
        idx = np.argwhere(np.abs(diff_list - offset) <= rad)
        idx_ = np.argwhere(np.abs(diff_list - offset) > rad)
        diff_list = diff_list[idx]
        out_ = np.append(out_, in_[idx_])
        missed += len(in_[idx_])
        if len(diff_list) == 0:
            return np.zeros(len(events_ref)), 1.0, 0
        else:
            return (
                np.squeeze(diff_list),
                missed / len(events_ref),
                len(out_) / len(times),
            )

    def _save_performance(self, errors, missed, excess):
        """
        stores performance metrics after detecting events in reference
        """
        self._errors = errors  # array of all errors
        self._mean_error = np.mean(errors)
        self._missed = missed
        self._excess = excess
        self._quantiles = self._compute_quantiles(errors)
        self._offset = self._compute_offset(self._mean_error, self._quantiles)

    def _save_parameters(self, parameters):
        """
        Saves the optimisation parameters for event detection
        """
        self._parameters = parameters

    @staticmethod
    def _compute_quantiles(errors: np.ndarray):
        """Computes the 2.5th and 97.5th percentile of an array of errors

        Args:
            errors: array of errors

        Returns:
            np.ndarray: Size 2 array containing the 2.5th and 97.5th percentile in this order
        """
        if errors.size == 0:
            return np.array([np.nan, np.nan])
        else:
            return np.array([np.percentile(errors, 2.5), np.percentile(errors, 97.5)])

    @staticmethod
    def _compute_offset(mean_error: float, quantiles: np.ndarray) -> float:
        """
        Computes the offset of this instance's performance
        """
        if quantiles[0] * quantiles[1] >= 0:
            return mean_error
        else:
            return 0

    @abstractmethod
    def optimise(self, signal: np.ndarray):
        """
        Optimizes the method parameters accroding to the reference

        Args:
            signal: signal according to time where events are detected
        """
        raise NotImplementedError


# TODO: remove this class (only there for testing)
class GrfTestEventDetection(BaseOptimisedEventDetection):
    def __init__(self, configs, context, label, offset, trial_ref=None):
        super().__init__(configs, context, label, offset, trial_ref)
        self.frate = 100  # TODO Hz --> take it from c3d file. How?

    def optimise(self, signal):
        pass

    def _get_range(self, group):
        if self._label == FOOT_STRIKE:
            return range(len(group) - 1)
        else:
            return range(len(group) - 1, 0, -1)

    def processing_masked_signal(self, grf_signal):
        nan_mask = np.isnan(grf_signal.data).astype(int)
        non_nan_groups = np.split(
            np.arange(len(grf_signal)), np.where(nan_mask[:-1] & ~nan_mask[1:])[0] + 1
        )

        min_duration = 60
        for group in non_nan_groups:
            if len(group) < min_duration:
                grf_signal[group] = np.nan

        nan_mask_ = np.isnan(grf_signal.data).astype(int)
        non_nan_groups_ = np.split(
            np.arange(len(grf_signal)), np.where(nan_mask_[:-1] & ~nan_mask_[1:])[0] + 1
        )

        zero_threshold = 5
        for group_ in non_nan_groups_:
            close_to_zero = np.abs(grf_signal[group_].data) < zero_threshold
            # print(grf_signal[group_].data)
            for i, j in enumerate(self._get_range(group_)):
                if close_to_zero[j] and close_to_zero[self._get_range(group_)[i + 1]]:
                    grf_signal[group_[j]] = np.nan
                elif np.isnan(grf_signal[group_].data[j]):
                    continue
                else:
                    break

        # fig, axs = plt.subplots(2, 1, figsize = (10, 8), sharex = True)
        # fig.tight_layout()
        # nan_mask__ = np.isnan(grf_signal.data).astype(int)
        # non_nan_groups__ = np.split(
        #     np.arange(len(grf_signal)), np.where(nan_mask__[:-1] & ~nan_mask__[1:])[0] + 1
        # )
        # max_length =  np.max([len(group__) for group__ in non_nan_groups__])
        # for l, group__ in enumerate(non_nan_groups__):
        #     signal_ = np.zeros(max_length)
        #     deriv_ = np.zeros(max_length)
        #     signal = grf_signal[group__].data
        #     signal = signal[~np.isnan(signal)]
        #     len_ = len(signal)
        #     if len_ > 0:
        #         signal_[-len_:] = signal
        #         deriv = (signal[1:] - signal[:-1])*self.frate
        #         deriv_[-len_+1:] = deriv
        #         axs[0].plot(signal_, label = l)
        #         axs[1].plot(deriv_)
        # axs[0].legend(loc="center left")
        # plt.show()
        return grf_signal

    def _detect_events(self, trial):
        print(f"\n\t {self._label} {self._context}\n")
        markers = trial.get_data(model.DataCategory.MARKERS)
        GRF_3d = (
            markers.sel(channel="LNormalisedGRF")
            if self._context == LEFT
            else markers.sel(channel="RNormalisedGRF")
        )
        GRF = GRF_3d.loc["z"]
        GRF_processed = self.processing_masked_signal(GRF)
        nan_mask = np.isnan(GRF_processed.data).astype(int)
        if self._label == FOOT_STRIKE:
            index = np.where((~nan_mask[1:]) & (nan_mask[:-1]))[0] + 1
        else:
            index = np.where((nan_mask[1:]) & (~nan_mask[:-1]))[0]
        time_ = GRF.time.data
        events = time_[index]
        return events


class PeakEventDetection(BaseOptimisedEventDetection, ABC):
    """Abstract class for event detectors that use peak detection of marker-based methods.

    This class provides a common interface for detecting events by finding peaks,
    which makes them interchangeable."""

    def optimise(self, signal):
        """
        Optimizes the method parameters accroding to the reference

        Args:
            signal: signal according to time where events are detected

        Returns:
            np.ndarray: indeces of each detected event after the optimization
            dict: optimization parameters for the method
        """

        def return_best(
            min_acc, acc, min_missed, missed, min_excess, excess, current_best, test
        ):
            acc_test = np.mean(acc)
            if missed < min_missed:
                return acc_test, missed, excess, test
            elif missed == min_missed:
                if excess < min_excess:
                    return acc_test, missed, excess, test
                elif excess == min_excess:
                    if np.abs(acc_test) < np.abs(min_acc):
                        return acc_test, missed, excess, test
                    else:
                        return min_acc, min_missed, min_excess, current_best
                else:
                    return min_acc, min_missed, min_excess, current_best
            else:
                return min_acc, min_missed, min_excess, current_best

        distances = (self.min_dist - np.arange(-0.2, 0.55, 0.05)) * self.frate
        prominences = np.arange(0.05, 0.75, 0.05)
        min_missed = 1.0
        min_accuracy = 100
        min_excess = 100
        best_params = {"distance": distances[0], "prominence": prominences[0]}
        i = 1
        for d in distances:
            i += 1
            for p in prominences:
                if d >= 1:
                    index, _ = sp.signal.find_peaks(-signal, distance=d, prominence=p)
                    if self.trial_ref is not None:
                        times = (
                            self.trial_ref.get_data(model.DataCategory.MARKERS)[
                                :, :, index
                            ]
                            .coords["time"]
                            .values
                        )
                    else:
                        raise ValueError("Reference trial must be provided")
                    acc, missed, excess = self._get_accuracy(times)
                    min_accuracy, min_missed, min_excess, best_params = return_best(
                        min_accuracy,
                        acc,
                        min_missed,
                        missed,
                        min_excess,
                        excess,
                        best_params,
                        {"distance": d, "prominence": p},
                    )
        idx, _ = sp.signal.find_peaks(
            -signal,
            distance=best_params["distance"],
            prominence=best_params["prominence"],
        )
        return idx, best_params

    def _detect_events(self, trial: model.Trial) -> np.ndarray:
        """Detects the events in the trial using peak detection.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the timings of the detected events.
        """

        if hasattr(self, "_parameters"):
            parameters = self._parameters
            distance = (
                parameters["distance"] if "distance" in parameters.keys() else None
            )
            prominence = (
                parameters["prominence"] if "prominence" in parameters.keys() else None
            )
            height = parameters["height"] if "height" in parameters.keys() else None
        else:
            parameters = None
        points = self._get_relevant_channels(trial)
        method = self._get_output(points)
        method = self.normalize(method)
        if parameters is None:
            index, parameters = self.optimise(method)
            self._save_parameters(parameters)
        else:
            index, _ = sp.signal.find_peaks(
                -method,
                distance=distance,
                prominence=prominence,
                height=height,
            )
        time = trial.get_data(model.DataCategory.MARKERS).coords["time"].values
        times = time[index[index < time.shape]]
        return times

    def _get_heel_or_toe(self, trial: model.Trial) -> xr.DataArray:
        """
        Returns the heel or toe marker trjectory, either left or right, according to the instance
        """
        if self._context == LEFT and self._label == FOOT_STRIKE:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_HEEL
            )
        elif self._context == LEFT and self._label == FOOT_OFF:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_TOE
            )
        elif self._context == RIGHT and self._label == FOOT_STRIKE:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_HEEL
            )
        else:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_TOE
            )
        return point

    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        """
        Performs and returns min-max normalisation of an array
        """
        min_pos = np.min(signal)
        max_pos = np.max(signal)
        norm_traj = (signal - min_pos) / (max_pos - min_pos)
        return norm_traj

    @abstractmethod
    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detecion
        """
        raise NotImplementedError

    @abstractmethod
    def _get_output(self, points: dict) -> np.ndarray:
        """
        Computes the ouput marker-based method on which peak detection is performed

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: output method that allows event detection through its peaks
        """
        raise NotImplementedError


class Zeni(PeakEventDetection):
    """
    Class for marker-based event detector based on Zeni et al. (2008)
    """

    _EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]
    _CODE = "Zen"

    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel or Toe, left or right according to the type and side of this instance, and sacrum
        """
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        point = self._get_heel_or_toe(trial)
        [sacrum, point] = self._rotate_markers([sacrum, point], trial)
        points = {}
        points["sacrum"] = sacrum
        points["point"] = point
        return points

    def _get_output(self, points: dict) -> np.ndarray:
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. the distance between the heel (or toe) and the sacrum

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: distance between the heel (or toe) and the sacrum
        """
        if "point" in points.keys() and "sacrum" in points.keys():
            point = points["point"]
            sacrum = points["sacrum"]
        else:
            raise ValueError(
                f"Sacrum or {'heel' if self._label == FOOT_STRIKE else 'toe'} trajectories have not been provided"
            )
        if self._label == FOOT_OFF:
            distance = sacrum - point
        else:
            distance = point - sacrum
        distance = self.normalize(distance.sel(axis="x"))
        return -distance.to_numpy()


class Desailly(PeakEventDetection):
    """
    Class for marker-based event detector based on Desailly et al. (2009)
    """

    _EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]
    _CODE = "Des"

    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel or Toe, left or right according to the type and side of this instance
        """
        point = self._get_heel_or_toe(trial)
        [point] = self._rotate_markers([point], trial)
        points = {}
        points["point"] = point
        return points

    def _get_output(self, points: dict):
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. output of the High Pass Algorithm

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: output of the High Pass Algorithm
        """
        if self.ref_events is None and not hasattr(self, "gait_freq"):
            raise ValueError("Reference trial should be provided")
        elif self.ref_events is not None and not hasattr(self, "gait_freq"):
            self.gait_freq = 1 / np.mean(self.ref_events[1:] - self.ref_events[:-1])
        if "point" in points.keys():
            point = points["point"]
        else:
            raise ValueError(
                f"{'Heel' if self._label == FOOT_STRIKE else 'Toe'} trajectory has not been provided"
            )
        point_norm = point.meca.normalize()
        filt_point = self.filt_signal(point_norm, 4, 7)
        point_hpfilt = self.sos_filt_signal(filt_point, 4, 0.5 * self.gait_freq, "high")
        if self._label == FOOT_OFF:
            return self.sos_filt_signal(point_hpfilt, 4, 1.1 * self.gait_freq, "high")[
                0, :
            ]
        else:
            return -point_hpfilt[0, :]

    def filt_signal(self, signal, order, Freq, type="low"):
        """
        Filters signals using a zero-lag butterworth filter

        Args:
            signal: array to be filtered
            order: order of the Butterworth filter
            Freq: cut-off frequency in Herz
            type: "low" for a low-pass filter and "high" for high-pass filter

        Returns:
            np.ndarray: filtered signal
        """
        b, a = sp.signal.butter(order, Freq, btype=type, fs=self.frate)
        return sp.signal.filtfilt(b, a, signal)

    def sos_filt_signal(self, signal, order, Freq, type):
        """
        Filters signals using a second-order section digital filter sos butterworth filter

        Args:
            signal: array to be filtered
            order: order of the Butterworth filter
            Freq: cut-off frequency in Herz
            type: "low" for a low-pass filter and "high" for high-pass filter

        Returns:
            np.ndarray: filtered signal
        """
        sos = sp.signal.butter(order, Freq, btype=type, output="sos", fs=self.frate)
        return sp.signal.sosfilt(sos, signal)


class AC(PeakEventDetection):
    """
    Factory class for all marker-based Auto-Correlation methods for event detection described in Fonseca et al. (2022)
    """

    _EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]
    _CODE = "AC"

    def __init__(self, configs, context, label, functions, trial_ref, offset=0):
        """Initializes a new instance of the AC class.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            functions: list of functions that compute the target values
            trial_ref: Trial to be used as reference
        """
        super().__init__(configs, context, label, offset, trial_ref)
        self.functions = functions

    @classmethod
    def get_AC1(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker and the horizontal distance between the sacrum and the heel to detect heel strikes
        """
        cls._EVENT_TYPES = [FOOT_STRIKE]
        cls._CODE = "AC1"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x],
            trial_ref=trial_ref,
        )

    @classmethod
    def get_AC2(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker, the horizontal distance between the sacrum and the heel, and the angle of the foot to detect heel strikes
        """
        cls._EVENT_TYPES = [FOOT_STRIKE]
        cls._CODE = "AC2"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x, cls.Foot_alpha],
            trial_ref=trial_ref,
        )

    @classmethod
    def get_AC3(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the hortizontal distance between the anterior hips and the horizontal distance between the sacrum and the heel to detect heel strikes
        """
        cls._EVENT_TYPES = [FOOT_STRIKE]
        cls._CODE = "AC3"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Hip_x, cls.Sacr_Heel_x],
            trial_ref=trial_ref,
        )

    @classmethod
    def get_AC4(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker, the horizontal distance between the sacrum and the heel, the angle of the foot and the horizontal distance between the anterior hips to detect heel strikes
        """
        cls._EVENT_TYPES = [FOOT_STRIKE]
        cls._CODE = "AC4"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x, cls.Foot_alpha, cls.Hip_x],
            trial_ref=trial_ref,
        )

    @classmethod
    def get_AC5(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the vertical componenent of the toe marker and the horizontal distance between the sacrum and the toe to detect toe offs
        """
        cls._EVENT_TYPES = [FOOT_OFF]
        cls._CODE = "AC5"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Toe_z, cls.Sacr_Toe_x],
            trial_ref=trial_ref,
        )

    @classmethod
    def get_AC6(cls, configs, context, label, trial_ref):
        """
        Initializes an instance of class AC using the vertical componenent of the toe marker, the horizontal distance between the sacrum and the toe, and the angle of the foot to detect toe offs
        """
        cls._EVENT_TYPES = [FOOT_OFF]
        cls._CODE = "AC6"
        return cls(
            configs,
            context,
            label,
            functions=[cls.Toe_z, cls.Sacr_Toe_x, cls.Foot_alpha],
            trial_ref=trial_ref,
        )

    @staticmethod
    def p_function(param: np.ndarray, param_ref: np.ndarray) -> np.ndarray:
        """Parameter function estimation: for any target value, the mean of the target value of the reference trial at the times of events is substracted

        Args:
            param: any target value (eg: heel trajectory in z direction)
            param_ref: same target value as param in the reference trial at times of true events

        Returns:
            np.ndarray: Parameter function estimation
        """
        return np.abs(param - np.mean(param_ref))

    def _get_relevant_channels(self, trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel, toe (left or right according to the instance), left and right anterior hips and sacrum
        """
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        l_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.L_ANT_HIP
        )
        r_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.R_ANT_HIP
        )
        if self._context == LEFT:
            toe = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_TOE
            )
            heel = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_HEEL
            )
        else:
            toe = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_TOE
            )
            heel = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_HEEL
            )
        points = {}
        points["sacrum"] = sacrum
        points["l_ant_hip"] = l_ant_hip
        points["r_ant_hip"] = r_ant_hip
        points["toe"] = toe
        points["heel"] = heel
        return points

    def _get_output(self, points: dict):
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. output of the Autocorrelation method using the instances's parameters

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: Autocorrelation output
        """
        points_ref = self._get_relevant_channels(self.trial_ref)
        gait_params = self._get_params(points)
        gait_params_ref = self._get_params(points_ref)
        p_ = np.zeros(len(gait_params[0]))
        for i, parameter in enumerate(gait_params_ref):
            params_ref_at_events = self._get_param_at_ref_events(parameter)
            p_ += self.p_function(gait_params[i], params_ref_at_events)
        return p_

    def _get_param_at_ref_events(self, param_ref: np.ndarray) -> np.ndarray:
        """Computes the mean of the target value obtained at the true events

        Args:
            param_ref : target value/parameter (eg. vetical component of the heel) of the reference trial

        Returns:
            np.ndarray : target values only at the times of the true events
        """
        if self.trial_ref is None or self.ref_events is None:
            raise ValueError("Reference trial should be provided")
        else:
            idx = np.isin(
                self.trial_ref.get_data(model.DataCategory.MARKERS).time.data,
                self.ref_events,
            )
            return param_ref[idx]

    def _get_params(self, points: dict) -> list[np.ndarray]:
        """Gets the target values/parameters for the autocorrelation calculation

        Args:
            points: dictionary containing the relevant markers trajectories

        Returns:
        list[np.ndarray] : list of target values/parameters
        """
        params = []
        for function in self.functions:
            params.append(function(points))
        return params

    @classmethod
    def Heel_z(cls, points: dict) -> np.ndarray:
        """
        Extracts the vertical position of the heel marker
        """
        if "heel" in points.keys():
            heel = points["heel"]
            return cls.normalize(heel.loc["z"])
        else:
            raise ValueError("Heel trajectory has not been provided")

    @classmethod
    def Toe_z(cls, points: dict) -> np.ndarray:
        """
        Extracts the vertical position of the toe marker
        """
        if "toe" in points.keys():
            toe = points["toe"]
            return cls.normalize(toe.loc["z"])
        else:
            raise ValueError("Toe trajectory has not been provided")

    @classmethod
    def Sacr_Heel_x(cls, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the heel and the sacrum
        """
        if "sacrum" in points.keys() and "heel" in points.keys():
            sacrum = points["sacrum"]
            heel = points["heel"]
            diff = np.abs(sacrum.loc["x"] - heel.loc["x"])
            return cls.normalize(diff)
        else:
            raise ValueError("Sacrum or heel trajectories have not been provided")

    @classmethod
    def Sacr_Toe_x(self, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the toe and the sacrum
        """
        if "sacrum" in points.keys() and "toe" in points.keys():
            sacrum = points["sacrum"]
            toe = points["toe"]
            diff = np.abs(sacrum.loc["x"] - toe.loc["x"])
            return self.normalize(diff)
        else:
            raise ValueError("Sacrum or toe trajectories have not been provided")

    @classmethod
    def Foot_alpha(cls, points: dict) -> np.ndarray:
        """
        Extracts the foot (defined by the heel and the toe) angle relative to the floor
        """
        if "heel" in points.keys() and "toe" in points.keys():
            heel = points["heel"]
            toe = points["toe"]
            angle = np.arctan2(
                toe.loc["z"] - heel.loc["z"],
                toe.loc["x"] - heel.loc["x"],
            )
            return cls.normalize(angle)
        else:
            raise ValueError("Heel or toe trajectories have not been provided")

    @classmethod
    def Hip_x(cls, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the anterior iliac spine markers
        """
        if "l_ant_hip" in points.keys() and "r_ant_hip" in points.keys():
            l_ant_hip = points["l_ant_hip"]
            r_ant_hip = points["r_ant_hip"]
            diff = np.abs(l_ant_hip.loc["x"] - r_ant_hip.loc["x"])
            return cls.normalize(diff)
        else:
            raise ValueError("Anterior hips trajectories have not been provided")


class EventDetector:
    """
    Container of Event detection methods to use for all sides and event types
    """

    def __init__(
        self,
        hs_left: BaseEventDetection,
        hs_right: BaseEventDetection,
        to_left: BaseEventDetection,
        to_right: BaseEventDetection,
    ):
        """Initializes a new instance of the EventDetector class.

        Args:
            hs_left: Event detection object class that predicts left Foot Strike
            hs_right: Event detection object class that predicts right Foot Strike
            to_left: Event detection object class that predicts left Foot Off
            to_right: Event detection object class that predicts right Foot Off
        """
        self.hs_left = hs_left
        self.hs_right = hs_right
        self.to_left = to_left
        self.to_right = to_right

    def detect_events(
        # TODO: add option to set own parameters
        self,
        trial: model.Trial,
    ) -> pd.DataFrame:
        """Detects events for all event types and sides

        Args:
            trial: The trial for which to detect the events.

        Returns:
            pd.DataFrame: Table containing all the detected events
        """
        hs_l_events = self.hs_left.detect_events(trial)
        hs_r_events = self.hs_right.detect_events(trial)
        to_l_events = self.to_left.detect_events(trial)
        to_r_events = self.to_right.detect_events(trial)

        events = self._create_data_frame(
            hs_l_events, hs_r_events, to_l_events, to_r_events
        )

        return events

    def _create_data_frame(
        self, hs_l_events, hs_r_events, to_l_events, to_r_events
    ) -> pd.DataFrame:
        """Creates a dataframe from the detected events.

        Args:
            hs_l_events: array of detected left foot strike timings
            hs_r_events: array of detected right foot strike timings
            to_l_events: array of detected left foot off timings
            to_r_events: array of detected right foot off timings

        Returns:
            pd.DataFrame: Complete table of detected events
        """
        hs_l_events_df = self.hs_left._create_data_frame(hs_l_events)
        hs_r_events_df = self.hs_right._create_data_frame(hs_r_events)
        to_l_events_df = self.to_left._create_data_frame(to_l_events)
        to_r_events_df = self.to_right._create_data_frame(to_r_events)

        events = pd.concat(
            [hs_l_events_df, hs_r_events_df, to_l_events_df, to_r_events_df]
        )
        events = events.sort_values(
            by=self.hs_left._TIME_COLUMN, ascending=True
        ).reset_index(drop=True)
        return events

    # @staticmethod
    # def add_events_to_trial(trial: model.Trial, events: pd.DataFrame):
    #     """
    #     Adds a table of events as a trial's attribute
    #     """
    #     trial_with_events = trial.events(events)
    #     return trial_with_events


class EventDetectorBuilder:
    """
    Mapping class to easily access Event detector classes
    """

    MAPPING = {
        "Zen": Zeni,
        "Des": Desailly,
        "AC1": AC.get_AC1,
        "AC2": AC.get_AC2,
        "AC3": AC.get_AC3,
        "AC4": AC.get_AC4,
        "AC5": AC.get_AC5,
        "AC6": AC.get_AC6,
        "GRF": GrfTestEventDetection,  # TODO: switch back to GRF2...
    }

    @classmethod
    def get_method(cls, name: str):
        """
        Gets event detection method from a string code
        """
        if name not in cls.MAPPING.keys():
            raise ValueError(f"Unknown name: {name}")
        return cls.MAPPING[name]

    @classmethod
    def get_event_detector(
        cls, configs: mapping.MappingConfigs, name: str, offset: float = 0, trial=None
    ) -> EventDetector:
        method = cls.get_method(name)
        return EventDetector(
            method(configs, LEFT, FOOT_STRIKE, offset, trial),
            method(configs, RIGHT, FOOT_STRIKE, offset, trial),
            method(configs, LEFT, FOOT_OFF, offset, trial),
            method(configs, RIGHT, FOOT_OFF, offset, trial),
        )

    @classmethod
    def get_mixed_event_detector(
        cls,
        configs: mapping.MappingConfigs,
        name_to_l: str,
        name_to_r: str,
        name_hs_l: str,
        name_hs_r: str,
        offset: float = 0,
        trial=None,
    ) -> EventDetector:
        method_to_l = cls.get_method(name_to_l)
        method_to_r = cls.get_method(name_to_r)
        method_hs_l = cls.get_method(name_hs_l)
        method_hs_r = cls.get_method(name_hs_r)
        return EventDetector(
            method_to_l(configs, LEFT, FOOT_OFF, offset, trial),
            method_to_r(configs, RIGHT, FOOT_OFF, offset, trial),
            method_hs_l(configs, LEFT, FOOT_STRIKE, offset, trial),
            method_hs_r(configs, RIGHT, FOOT_STRIKE, offset, trial),
        )


class AutoEventDetection:
    """
    Class for the automatisation of marker-based event detection
    """

    def __init__(
        self,
        configs,
        trial_ref,
        method_list=[
            Zeni,
            Desailly,
            AC.get_AC1,
            AC.get_AC2,
            AC.get_AC3,
            AC.get_AC4,
            AC.get_AC5,
            AC.get_AC6,
        ],
    ):
        """Initializes a new instance of the AutoEventDetection class.

        Args:
            configs: The mapping configurations.
            trial_ref: Trial to be used as reference
            method_list: list of methods (classes) to attempt detection of events on the reference
        """
        self._configs = configs
        self.trial_ref = trial_ref
        self.method_list = method_list

    def get_optimised_event_detectors(self) -> EventDetector:
        """Performs event detection using all the specified methods on the reference trial, and selects the best performing one

        Returns:
            EventDetector: instance of EventDetector class containing the optimized event detectors for each event type and side
        """
        # TODO: optimise cut trial
        # TODO: make function shorter
        opt_detectors: dict[str, dict] = {FOOT_STRIKE: {}, FOOT_OFF: {}}
        event_detectors_ = []  # TODO: remove (here only for testing)
        for label in EVENT_TYPES:
            # print(label)
            for context in SIDES:
                # print(context)
                opt = np.zeros(len(self.method_list))
                event_detectors = []
                for idx, method in enumerate(self.method_list):
                    event_detector = method(
                        self._configs, context, label, trial_ref = self.trial_ref
                    )  ##an instance for each side
                    if label in event_detector._EVENT_TYPES:
                        times = event_detector._detect_events(self.trial_ref)
                        errors, missed, excess = event_detector._get_accuracy(times)
                        event_detector._save_performance(errors, missed, excess)
                        event_detectors.append(event_detector)
                        event_detectors_.append(
                            event_detector
                        )  # TODO: remove (here only for testing)
                        opt[idx] = self._optim_function(event_detector)
                index = np.argmax(opt)
                opt_detectors[label][context] = event_detectors[index]
        event_detector = EventDetector(
            opt_detectors[FOOT_STRIKE][LEFT],
            opt_detectors[FOOT_STRIKE][RIGHT],
            opt_detectors[FOOT_OFF][LEFT],
            opt_detectors[FOOT_OFF][RIGHT],
        )
        self.event_detectors = event_detectors_  # TODO: remove (here only for testing)
        return event_detector

    def _optim_function(self, detector: BaseOptimisedEventDetection) -> float:
        """Computes the result of the optimisation function for a specific event detector

        Args:
            detector: Event detector whose performance on the reference is being evaluated

        Returns:
            float: result of the optimisation function
        """
        shifted_accuracy = detector._mean_error - detector._offset
        quantiles = detector._quantiles
        width = np.abs(quantiles[1] - quantiles[0])
        return (
            0.6 * (1 - np.abs(shifted_accuracy))
            + 0.4 * (1 - width)
            + 0.5 * (1 - detector._missed)
            + 0.5 * (1 - detector._excess)
        )

    ##TODO: TO REMOVE (only there for testing)
    def plot_accuracies(self):
        nb_methods = len(self.event_detectors)
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        fig.tight_layout()
        x_ = range(nb_methods)
        ax[0].hlines(y=0, xmin=0, xmax=np.max(x_), colors="black")
        ax[0].scatter(
            x=range(nb_methods),
            y=[self.event_detectors[i]._mean_error for i in range(nb_methods)],
            marker="_",
            color="tab:blue",
        )
        ax[0].scatter(
            x=range(nb_methods),
            y=[self.event_detectors[i]._quantiles[0] for i in range(nb_methods)],
            marker="2",
            color="tab:blue",
        )
        ax[0].scatter(
            x=range(nb_methods),
            y=[self.event_detectors[i]._quantiles[1] for i in range(nb_methods)],
            marker="1",
            color="tab:blue",
        )
        ax[0].vlines(
            x=range(nb_methods),
            ymin=[self.event_detectors[i]._quantiles[0] for i in range(nb_methods)],
            ymax=[self.event_detectors[i]._quantiles[1] for i in range(nb_methods)],
            colors="tab:blue",
        )

        width_bar = 0.8
        y_m = np.array([self.event_detectors[i]._missed for i in range(nb_methods)])
        y_e = np.array([self.event_detectors[i]._excess for i in range(nb_methods)])
        for k, (y_, m_type) in enumerate(zip([y_m, y_e], ["Missed", "Excess"])):
            ax[k + 1].bar(x=x_, height=y_ * 100, width=width_bar)
            for i in range(len(x_)):
                ax[k + 1].text(
                    x=x_[i] - 0.4 * width_bar,
                    y=y_[i] * 100,
                    s=str(np.round(y_[i] * 100, decimals=1)) + "%",
                    fontsize="medium",
                )

            max = (
                np.max(
                    [
                        self.event_detectors[i]._missed
                        if m_type == "Missed"
                        else self.event_detectors[i]._excess
                        for i in range(nb_methods)
                    ]
                )
                * 100
            )
            ax[k + 1].set_ylim([0, max])
            ax[k + 1].set_title(
                f"Percentage of {m_type} detected events for each method"
            )
            ax[k + 1].set_ylabel(m_type + " detected events")
            if max < 20:
                ax[k + 1].set_yticks(
                    np.arange(0, max + 2, 2),
                    [str(int(k)) + "%" for k in np.arange(0, max + 2, 2)],
                )
            else:
                ax[k + 1].set_yticks(
                    np.arange(0, np.round(max + 20, decimals=20), 20),
                    [str(int(k)) + "%" for k in np.arange(0, max + 20, 20)],
                )
        ax[0].set_yticks(np.arange(-0.5, 0.6, 0.1))
        ax[0].set_ylim([-0.5, 0.5])
        ax[0].set_ylabel("Error [s]")
        ax[0].set_title("Error of event detection for each method")
        ax[0].grid()
        ax[2].set_xticks(
            x_,
            [
                self.event_detectors[i]._label
                + " "
                + self.event_detectors[i]._CODE
                + " "
                + self.event_detectors[i]._context
                for i in range(nb_methods)
            ],
            rotation=90,
        )
        plt.show()
