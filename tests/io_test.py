import functools
import multiprocessing
import os
import shutil
import time
from typing import Any, Dict, Generator, Iterable, List, Tuple

import pytest

from rgnn_at_scale.helper.io import Storage


cache_base = 'cache_io_test'


def chunk(iterable: Iterable, chunk_size: int = 10) -> Generator[List[Any], Iterable, None]:
    next_chunk = []
    i_chunk = 0
    for item in iterable:
        next_chunk.append(item)
        i_chunk += 1
        if i_chunk == chunk_size:
            yield next_chunk
            next_chunk = []
    if len(next_chunk):
        yield next_chunk


def store(params: Dict[str, Any], artifact: Dict[str, Any], table: str, cache_dir: str):
    storage = Storage(cache_dir)
    return storage.save_artifact(table, params, artifact)


def chunked_store(params_and_artifact: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                  table: str, cache_dir: str, sleep: float = 0.1):
    storage = Storage(cache_dir)
    for params, artifact in params_and_artifact:
        storage.save_artifact(table, params, artifact)
        time.sleep(sleep)
    # Check that all artifacts of process are available
    for params, artifact in params_and_artifact:
        assert storage.load_artifact(table, params) == artifact
    # Check that also artifacts of other process are available
    storage_new = Storage(cache_dir)
    elements_new = storage_new.find_artifacts(table, {})
    elements = storage.find_artifacts(table, {})
    assert len(elements_new) <= len(elements)


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

        _ = storage.save_artifact(table, params1, artifact1)
        _ = storage.save_artifact(table, params2, artifact2)

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

    def test_consecutive_multiprocessed_writes(self):
        table = 'test'
        n_elements = 100
        n_processes = 10
        chunk_size = 10
        sleep = 0.1
        cache_dir = os.path.join(cache_base, self.test_locked_write.__name__)

        params_list = [{'a': i, 'b': i**2} for i in range(n_elements)]
        artifact_list = [{'c': i + 3} for i in range(n_elements)]

        pool = multiprocessing.Pool(n_processes)
        pool.map(functools.partial(chunked_store, table=table, cache_dir=cache_dir, sleep=sleep),
                 chunk(zip(params_list, artifact_list), chunk_size))

        storage = Storage(cache_dir)
        assert len(storage.find_artifacts(table, {})) == n_elements

        for params, artifact in zip(params_list, artifact_list):
            storage.load_artifact(table, params) == artifact
