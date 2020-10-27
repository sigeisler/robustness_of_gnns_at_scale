import functools
import multiprocessing
import os
import shutil

import pytest

from rgnn.io import Storage


cache_base = 'cache_test'


def store(params, artifact, table, cache_dir):
    storage = Storage(cache_dir)
    return storage.save_artifact(table, params, artifact)


@pytest.fixture
def cleandir():
    if os.path.isdir(cache_base):
        shutil.rmtree(cache_base)
    yield
    shutil.rmtree(cache_base)
    os.makedirs(cache_base)
    open(os.path.join(cache_base, '.gitkeep'), 'a').close()


@pytest.mark.usefixtures("cleandir")
class TestStorage():

    def test_save_and_retrieve_artifacts(self):
        table = 'test'
        params1 = {'a': 1, 'b': {'c': 3}}
        artifact1 = {'d': 1}
        params2 = {'a': 2, 'b': {'c': 3}}
        artifact2 = {'d': 4}

        storage = Storage(os.path.join(cache_base, self.test_save_and_retrieve_artifacts.__name__))

        id1 = storage.save_artifact(table, params1, artifact1)
        id2 = storage.save_artifact(table, params2, artifact2)

        assert storage.load_artifact(table, params1) == artifact1
        assert storage.load_artifact(table, params2) == artifact2

        assert len(storage.find_artifacts(table, {'a': 3})) == 0
        temp_documents = storage.find_artifacts(table, {'a': 1})
        assert len(temp_documents) == 1
        assert temp_documents[0]['params'] == params1
        assert temp_documents[0]['artifact'] == artifact1
        assert len(storage.find_artifacts(table, {'b': {'c': 3}})) == 2

    def test_locked_write(self):
        table = 'test'
        n_processes = 100
        cache_dir = os.path.join(cache_base, self.test_locked_write.__name__)

        params_list = [{'a': i, 'b': i**2} for i in range(n_processes)]
        artifact_list = [{'c': i + 3} for i in range(n_processes)]

        pool = multiprocessing.Pool(n_processes)
        pool.starmap(functools.partial(store, table=table, cache_dir=cache_dir),
                     zip(params_list, artifact_list))

        storage = Storage(cache_dir)
        assert len(storage.find_artifacts(table, {})) == n_processes

        for params, artifact in zip(params_list, artifact_list):
            storage.load_artifact(table, params) == artifact
