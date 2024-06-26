from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from typing import Union, Tuple


class CLONAX:

    def __init__(self, generations: int = 8,
                 memory_size: int = 120,
                 remaining_ratio: float = 0.1,
                 replaceable_size_ratio: float = 0.5,
                 n_to_clone: int = 20,
                 n_best_clones: int = 10,
                 n_antigens_to_average: int = 5,
                 with_proportion: bool = True):

        # parameters mentioned in the source paper
        self.generations = generations
        self.memory_size = memory_size  # m
        self.remaining_size = max(1, int(self.memory_size * remaining_ratio))  # r
        self.n_replaceable_size = max(1, int(self.remaining_size * replaceable_size_ratio))  # d
        self.n_to_clone = n_to_clone  # n
        self.n_best_clones = n_best_clones  # k
        self.n_antigens_to_average = n_antigens_to_average  # p

        self.with_proportion = with_proportion

        # parameters initialized during fit
        self.training_set = None
        self.training_labels = None
        self.n_features = None

        self._class_0_memory = None
        self._class_1_memory = None
        self.remaining_population = None

        self.remaining_population_labels = None

    @property
    def class_0_memory(self) -> np.ndarray:
        return self._class_0_memory

    @class_0_memory.setter
    def class_0_memory(self, vals: np.ndarray):
        self._class_0_memory = vals

    @property
    def class_1_memory(self) -> np.ndarray:
        return self._class_1_memory

    @class_1_memory.setter
    def class_1_memory(self, vals: np.ndarray):
        self._class_1_memory = vals

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray) -> CLONAX:
        """
        Fit CLONAX classifier from the training dataset.

        Parameters
        ----------
        X_train : numpy ndarray of shape (n_samples, n_features)
            Training data.
        y_train : numpy ndarray of shape (n_samples, )
            Target classes.
        Returns
        -------
        self : CLONAX
            The fitted CLONAX classifier.
        """
        # Init class parameters
        self.training_set = X_train.copy()
        self.training_labels = y_train.copy()
        self.n_features = X_train.shape[1]

        # Init population
        self._init_population()
        print('Population initialized')

        for generation_val in range(self.generations):

            idx_left = [i for i in range(len(self.training_set))]

            while len(idx_left) != 0:
                print(f'indexes left: {len(idx_left)}')
                # 2. Release Antigen
                # Choose random training element (antigen) without replacement and determine its class
                index = np.random.choice(idx_left)
                training_obj = self.training_set[index]
                training_obj_label = self.training_labels[index]

                if training_obj_label == 0:
                    same_class_memory_cells = self.class_0_memory
                else:
                    same_class_memory_cells = self.class_1_memory
                population = np.concatenate((same_class_memory_cells, self.remaining_population), axis=0)
                population_labels = np.array(([training_obj_label] * len(same_class_memory_cells) +
                                              self.remaining_population_labels.tolist())).reshape((-1, 1))

                # 3. Affinity
                # Calc distances and return best population to clone
                best_from_population, affinities, population_labels_sorted = self._check_n_best(training_obj,
                                                                                                population,
                                                                                                population_labels)

                # 4. Clone antibodies
                # Clone population and add Gaussian noise
                cloned_population, cloned_population_labels, rank = self._create_clones(best_from_population,
                                                                                        population_labels_sorted,
                                                                                        self.n_to_clone)

                # 5. Affinity maturation
                cloned_noisy_population = self._mutate(cloned_population, rank)

                # 6. Select clones
                cloned_noisy_population, _, cloned_population_labels = self._check_n_best(training_obj,
                                                                                          cloned_noisy_population,
                                                                                          cloned_population_labels)
                k_highest_affinity_clones = cloned_noisy_population[:self.n_best_clones, :]
                k_highest_affinity_clones_labels = cloned_population_labels[:self.n_best_clones]

                # 7. Average affinity
                clones_avg_affinities = self._average_affinities(k_highest_affinity_clones,
                                                                 training_obj_label)
                # 8. Filter noise
                clone_to_save_flag, idx_training_data_to_remove = \
                    self._filter_noise(k_highest_affinity_clones,
                                       k_highest_affinity_clones_labels,
                                       clones_avg_affinities)

                k_highest_affinity_clones = k_highest_affinity_clones[clone_to_save_flag]
                k_highest_affinity_clones_labels = k_highest_affinity_clones_labels[clone_to_save_flag]
                # clones_avg_affinities = clones_avg_affinities[clone_to_save_flag]
                idx_left = self._remove_from_training_set(idx_training_data_to_remove, idx_left, index)

                # 9. Update memory
                # average affinity or highest?
                if len(k_highest_affinity_clones) != 0:
                    self._update_memory(k_highest_affinity_clones,
                                        k_highest_affinity_clones_labels,
                                        training_obj)

                # 10. Update antibody repertoire
                self._update_remaining_population(training_obj)
        return self

    def _average_affinities(self,
                            k_highest_affinity_clones: np.ndarray,
                            training_obj_label: np.ndarray) -> np.ndarray:
        p = self.n_antigens_to_average
        affinities = self._check_affinities(self.training_set, k_highest_affinity_clones)
        label = training_obj_label

        same_class_p_nearest_mean_affinities: list[list[float]] = []
        for idx, clone in enumerate(k_highest_affinity_clones):
            # average affinities for p-nearest
            same_class_p_nearest_affinity = np.sort(affinities[self.training_labels == label, idx])[:p]
            same_class_p_nearest_mean_affinities += [[np.sum(same_class_p_nearest_affinity) / p]]
        return np.array(same_class_p_nearest_mean_affinities)

    def _check_n_best(self,
                      training_data: np.ndarray,
                      population: np.ndarray,
                      population_labels: np.ndarray,
                      dist_measure: distance = distance.euclidean) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate distance between training set and generated population using scipy.spatial.distance functions, and
        sort population by that distance from nearest to farthest, after that return self.n_to_clone nearest.

        Parameters
        ----------
        training_data: np.ndarray
            Training dataset.
        population: np.ndarray
            Generated population.
        population_labels: np.ndarray
            Generated population labels.
        dist_measure: distance = distance.euclidean
            Distance measure from scipy.spatial.distance.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The best self.n_to_clone cases from population. The first argument refers to the value of individuals from
            the population, the second to the value of the distance metric, and the third returns the class of each
            case.
        """
        # number of ABc + ABr objects
        n_population_obj = len(population)

        distances = np.array([dist_measure(training_data, population[j, :]) for j in range(n_population_obj)])

        # sort by scores
        population_with_scores = np.c_[population, distances]
        population_with_scores_sorted = population_with_scores[population_with_scores[:, -1].argsort()]
        population_labels_sorted = population_labels[population_with_scores[:, -1].argsort()]

        return population_with_scores_sorted[:self.n_to_clone, :-1], \
            population_with_scores_sorted[:self.n_to_clone, -1], \
            population_labels_sorted

    def _check_affinities(self,
                          training_data: np.ndarray,
                          population: np.ndarray,
                          dist_measure: distance = distance.euclidean) -> np.ndarray:
        """
        Returns values of the affinities between training_data and population. Affinity is calculated by using
        scipy.spatial.distance between training data individual and population individual, divided by number of the
        self.n_features.

        Parameters
        ----------
        training_data: np.ndarray
            Training dataset.
        population: np.ndarray
            Generated population.
        dist_measure: distance = distance.euclidean
            Distance measure from scipy.spatial.distance.

        Returns
        -------
        distances: np.ndarray
            Distances between training_data and population.

        """
        n_population_obj = len(population)
        n_training_data = len(training_data)

        distances = np.zeros((n_training_data, n_population_obj))
        for i in range(n_training_data):
            for j in range(n_population_obj):
                distances[i, j] = dist_measure(training_data[i, :], population[j, :]) / self.n_features
        return distances

    @staticmethod
    def _create_clones(population: np.ndarray,
                       population_labels: np.ndarray,
                       n_to_clone: int) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Creates new population by clonning existing population individual. The number of clones is defined in such a way
        that the individual of the population at the first index is cloned self.n_to_clone times, the last one starts
        not to be cloned, every individual between is cloned round(self.n_to_clone / (i + 1)) times, where i is the
        index of the individual.

        Parameters
        ----------
        population: np.ndarray
            Generated population.
        population_labels: np.ndarray
            Generated population labels.

        Returns
        -------
        cloned_population_arr, cloned_labels_arr, rank: Tuple[np.ndarray, np.ndarray, list]
            Returns an array of the cloned population, the labels of each individual and the cloned positions in a
            ranking (where the most similar have 1st place, etc.).
        """
        cloned_population: list[list[int]] = []
        cloned_labels: list[list[int]] = []
        rank: list[int] = []

        for i in range(min(len(population), n_to_clone)):
            num_of_copies = round(n_to_clone / (i + 1))

            cloned_population.extend([population[i, :].tolist()] * num_of_copies)
            cloned_labels.extend([population_labels[i].tolist()] * num_of_copies)
            rank.extend([i] * num_of_copies)
        cloned_population_arr = np.array(cloned_population)
        cloned_labels_arr = np.array(cloned_labels)

        return cloned_population_arr, cloned_labels_arr, rank

    def _init_population(self) -> None:
        """
        Initiates initial populations for the algorithm. It creates self.remaining_population,
        self.remaining_population_labels and self.class_0_memory, self.class_1_memory. Parameter self.with_proportion
        controls whether memory individuals ot the same proportion as those present in the training set are to be
        created.
        """

        if self.with_proportion:
            class_0_lst: list[list[int]] = []
            class_1_lst: list[list[int]] = []

            proportion_ratio = len(self.training_labels[self.training_labels == 0]) / len(self.training_labels)

            n_class_0 = int(proportion_ratio * self.memory_size)
            n_class_1 = self.memory_size - n_class_0

            while n_class_0 > 0 or n_class_1 > 0:
                obj = np.random.random((1, self.n_features))
                # obj /= np.sum(obj, axis=1)
                obj = obj.tolist()

                pred_label = self._classify(self.training_set, obj, self.training_labels)
                if pred_label == 0 and n_class_0 != 0:
                    class_0_lst += obj
                    n_class_0 -= 1
                elif pred_label == 1 and n_class_1 != 0:
                    class_1_lst += obj
                    n_class_1 -= 1

            self.class_0_memory = np.array(class_0_lst)
            self.class_1_memory = np.array(class_1_lst)

        remaining_population = np.random.random((self.remaining_size, self.n_features))
        scaler = MinMaxScaler()
        remaining_population = scaler.fit_transform(remaining_population)

        # Assign class using trained model
        remaining_population_labels = self._classify(self.training_set, remaining_population, self.training_labels)

        self.remaining_population = remaining_population
        self.remaining_population_labels = remaining_population_labels

    @staticmethod
    def _classify(X_train: Union[np.ndarray, pd.DataFrame],
                  X_test: Union[np.ndarray, pd.DataFrame],
                  y_train: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Classify test set individuals based on newly created population using KNN classification algorithm.

        Parameters
        ----------
        X_train: Union[np.ndarray, pd.DataFrame]
        X_test: Union[np.ndarray, pd.DataFrame]
        y_train: Union[np.ndarray, pd.DataFrame]

        Returns
        -------
        y_pred: np.ndarray
        """
        model = KNeighborsClassifier(n_neighbors=10, weights='distance').fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred

    def _mutate(self,
                clones: np.ndarray,
                rank: list) -> np.ndarray:
        """
        For each individual add random noise based on the position in a ranking where the most similar
        have 1st place, etc.).

        Parameters
        ----------
        clones: np.ndarray
        rank: list

        Returns
        -------
        mutated_clones: np.ndarray
            Clones population modified by noise.
        """
        # min_val + ((max_val - min_val) / (n_intervals - 1)) * iteration
        noise_variability = (1 - 1 / 100) / (self.n_to_clone - 1)
        noise_std = np.multiply(rank, noise_variability) + 1/100

        mutated_clones = clones.copy()
        random_noise = [np.random.normal(0, noise_variability, size=(clones.shape[1], ))
                        for noise_variability in noise_std]
        mutated_clones = mutated_clones + random_noise

        return mutated_clones

    def _filter_noise(self,
                      k_highest_affinity_clones: np.ndarray,
                      k_highest_affinity_clones_labels: np.ndarray,
                      clones_avg_affinities: np.ndarray) -> Tuple[list, list]:
        """
        Flags data to save and indexes of elements to remove based on the filtering rules defined in the source paper.

        Parameters
        ----------
        k_highest_affinity_clones: np.ndarray
        k_highest_affinity_clones_labels: np.ndarray
        clones_avg_affinities: np.ndarray

        Returns
        -------
        clone_to_save_flag, idx_training_data_to_remove: Tuple[list, list]
        """
        affinities = self._check_affinities(self.training_set, k_highest_affinity_clones)

        clone_to_save_flag: list[bool] = []
        idx_training_data_to_remove: list[int] = []

        for idx, affinity in enumerate(clones_avg_affinities):
            label = k_highest_affinity_clones_labels[idx]
            filters_for_affinities = np.logical_and(self.training_labels != label,
                                                    affinity > affinities[:, idx])

            n_smaller_than_avg = sum(filters_for_affinities)
            if n_smaller_than_avg >= 2:
                # remove == False
                clone_to_save_flag += [False]
            elif n_smaller_than_avg == 1:
                idx_training_data_to_remove += [i for i, x in enumerate(filters_for_affinities) if x]
                clone_to_save_flag += [True]
            else:
                clone_to_save_flag += [True]

        return clone_to_save_flag, idx_training_data_to_remove

    def _remove_from_training_set(self,
                                  idx_training_data_to_remove: list,
                                  idx_left: list,
                                  index: int) -> object:
        """
        Removes data from self.training_set and self.training_labels and returns indexes of the individuals which left.

        Parameters
        ----------
        idx_training_data_to_remove: list
            Indexes of the individuals to remove from self.training_set and self.training_labels.
        idx_left: list
            Indexes left after removal.
        index: int
            Index of the object under study.
        Returns
        -------
        idx_left: object
            Indexes of the individuals not removed from training dataset.
        """
        idx_left_ = idx_left.copy()

        self.training_set = np.delete(self.training_set, idx_training_data_to_remove, axis=0)
        self.training_labels = np.delete(self.training_labels, idx_training_data_to_remove, axis=0)

        idx_left_.remove(index)
        idx_left_ = np.array(idx_left_)
        idx_left_[idx_left_ > index] -= 1
        for idx_to_remove in idx_training_data_to_remove:
            if idx_to_remove in idx_left_:
                idx_left_ = np.delete(idx_left_, idx_to_remove)
            idx_left_[idx_left_ > idx_to_remove] -= 1
            idx_left_ = np.unique(idx_left_)
        idx_left_ = idx_left_.tolist()

        return idx_left_

    def _calc_dist(self,
                   training_obj: np.ndarray,
                   objects: np.ndarray,
                   dist_measure: distance = distance.euclidean) -> np.ndarray:
        """
        Calculates distances between training object under study and array of objects.

        Parameters
        ----------
        training_obj: np.ndarray
        objects: np.ndarray

        Returns
        -------
        objs_distances_arr: np.ndarray

        """
        n_objs = len(objects)
        objs_distances_arr = np.zeros((n_objs, 1))

        for j in range(n_objs):
            objs_distances_arr[j, :] = dist_measure(training_obj, objects[j, :]) \
                                       / self.n_features
        return objs_distances_arr

    def _update_memory(self,
                       k_highest_affinity_clones: np.ndarray,
                       k_highest_affinity_clones_labels: np.ndarray,
                       training_obj: np.ndarray) -> None:
        """
        Modifies self.class_0_memory and self.class_1_memory by assigning the best affinity clone to on of that set if
        that clone affinity is better than worst memory cell affinity.

        Parameters
        ----------
        k_highest_affinity_clones: np.ndarray
        k_highest_affinity_clones_labels: np.ndarray
        training_obj: np.ndarray
        """
        clones_distances = self._calc_dist(training_obj, k_highest_affinity_clones)

        idx_clones_sort = clones_distances.argsort(axis=0)
        k_highest_affinity_clones = np.take_along_axis(k_highest_affinity_clones, idx_clones_sort, axis=0)
        k_highest_affinity_clones_labels = np.take_along_axis(k_highest_affinity_clones_labels, idx_clones_sort, axis=0)
        clones_distances = np.take_along_axis(clones_distances, idx_clones_sort, axis=0)

        best_clone = k_highest_affinity_clones[0, :]
        best_clone_affinity = clones_distances[0]
        best_clone_label = k_highest_affinity_clones_labels[0]

        memory_objs = None
        if best_clone_label == 0:
            memory_objs = self.class_0_memory
        elif best_clone_label == 1:
            memory_objs = self.class_1_memory

        memory_objs_distances = self._calc_dist(training_obj, memory_objs)

        idx_memory_objs = memory_objs_distances.argsort(axis=0)
        memory_objs = np.take_along_axis(memory_objs, idx_memory_objs, axis=0)
        memory_objs_distances = np.take_along_axis(memory_objs_distances, idx_memory_objs, axis=0)
        worst_memory_cell_affinity = memory_objs_distances[-1]

        if best_clone_affinity < worst_memory_cell_affinity:
            memory_objs = memory_objs[:-1, :]
            memory_objs = np.append(memory_objs, best_clone.reshape((1, self.n_features)), axis=0)

            if best_clone_label == 0:
                self.class_0_memory = memory_objs
            elif best_clone_label == 1:
                self.class_1_memory = memory_objs

    def _update_remaining_population(self,
                                     training_obj: np.ndarray) -> None:
        """
        Recreate self.remaining_population by adding newly generated random individuals to some number of existing
        individuals with the best affinity to individual under study. Then classify them based on whole training dataset
         to get labels.

        Parameters
        ----------
        training_obj: np.ndarray
        """
        remaining_objs = self.remaining_population
        remaining_objs_distances = self._calc_dist(training_obj, remaining_objs)

        # sort array by distances
        idx_remaining_objs = remaining_objs_distances.argsort(axis=0)
        remaining_objs = np.take_along_axis(remaining_objs, idx_remaining_objs, axis=0)

        # create new individuals using random and scale them
        remaining_new_objs = np.random.random((self.n_replaceable_size, self.n_features))
        scaler = MinMaxScaler()
        remaining_new_objs = scaler.fit_transform(remaining_new_objs)

        # select some number of the highest affinity individuals from current remaining population and a
        n_to_leave = np.min(len(remaining_objs) - self.n_replaceable_size, 0)
        remaining_objs = remaining_objs[:n_to_leave, :]
        remaining_population = np.concatenate((remaining_objs, remaining_new_objs))

        # Assign class using trained model
        remaining_population_labels = self._classify(self.training_set, remaining_population, self.training_labels)

        self.remaining_population = remaining_population
        self.remaining_population_labels = remaining_population_labels

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X_test: np.ndarray
            Test samples of the same shape as training dataset.
        Returns
        -------
        y_pred: np.ndarray
            Class labels for each data sample.
        """

        X_train = np.concatenate((self.class_0_memory, self.class_1_memory))
        y_train = np.array([[0]] * len(self.class_0_memory) + [[1]] * len(self.class_1_memory)).reshape((-1,))

        y_pred = self._classify(X_train, X_test, y_train)

        return y_pred
