import pytest

import numpy as np

from classifiers.clonax import CLONAX


@pytest.fixture
def create_default_obj():
    print("setup default CLONAX class")
    yield CLONAX()
    print("teardown default CLONAX class")


@pytest.fixture
def f_create_clones_resource_1():
    print("set up create_clones_resource_1")

    population = np.array([[1, 1, 1],
                           [2, 2, 2],
                           [3, 3, 3]]
                          )
    population_labels = np.array([[0],
                                  [1],
                                  [0]]
                                 )
    n_to_clone = 3
    yield (population,
           population_labels,
           n_to_clone)

    print("tear down create_clones_resource_1")


@pytest.fixture
def f_create_clones_expected_1():
    print("set up create_clones_expected_1")
    cloned_population_arr = np.array([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1],
                                      [2, 2, 2],
                                      [2, 2, 2],
                                      [3, 3, 3]],
                                     )
    cloned_labels_arr = np.array([[0],
                                  [0],
                                  [0],
                                  [1],
                                  [1],
                                  [0]])
    rank = [0, 0, 0, 1, 1, 2]
    yield (cloned_population_arr,
           cloned_labels_arr,
           rank)

    print("tear down create_clones_expected_1")


class TestCLONAX:

    # @pytest.mark.parametrize('population,population_labels,expected',
    #                          ([[1, 1, 1], [2, 2, 2], [3, 3, 3]], [0, 1, 0,])
    def test_create_clones(self,
                           create_default_obj,
                           f_create_clones_resource_1,
                           f_create_clones_expected_1):
        clonax_inst = create_default_obj
        population, population_labels, n_to_clone = f_create_clones_resource_1
        cloned_population_arr, cloned_labels_arr, rank = f_create_clones_expected_1

        cloned_population_arr_res, cloned_labels_arr_res, rank_res = (
            clonax_inst._create_clones(population, population_labels, n_to_clone))

        np.testing.assert_array_equal(cloned_population_arr, cloned_population_arr_res)
        np.testing.assert_array_equal(cloned_labels_arr, cloned_labels_arr_res)
        np.testing.assert_array_equal(rank, rank_res)

