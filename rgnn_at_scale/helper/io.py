"""To handle the model/artifact zoo.
"""

from datetime import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Iterable, Mapping

from filelock import SoftFileLock
from sacred import Experiment
from tinydb import Query, TinyDB
from tinydb_serialization import SerializationMiddleware, Serializer
import torch
import scipy.sparse as sp
import numpy as np
import logging

from rgnn_at_scale.models import create_model, MODEL_TYPE


class DateTimeSerializer(Serializer):
    """TinyDB helper to serialize datetime.
    """
    OBJ_CLASS = datetime  # The class this serializer handles

    def encode(self, obj):
        return obj.strftime('%Y-%m-%dT%H:%M:%S')

    def decode(self, s):
        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')


class Storage():
    """Manages the storage of artifacts for later reuse.

    Parameters
    ----------
    cache_dir : str, optional
        Directory where the files shall be cached, by default 'cache'.
    experiment : sacred.Experiment, optional
        Experiment to extract meta information, by default None.
    lock_timeout : int, optional
        Timeout for a write lock of the local file db, by default 10.
    """

    def __init__(self, cache_dir: str = 'cache', experiment: Optional[Experiment] = None, lock_timeout: int = 10):
        self.experiment = experiment
        self.cache_dir = cache_dir
        self.lock_timeout = lock_timeout

        os.makedirs(cache_dir, exist_ok=True)

        self.dbs: Dict[str, TinyDB] = {}

    @staticmethod
    def locked_call(callable: Callable[[], Any], lock_file: str, lock_timeout: int) -> Any:
        """Locks a callable execution with a given timout and a specified lock file.

        Parameters
        ----------
        callable : Callable[[], Any]
            Whatever should be executed thread safe in a multi host environment.

        Returns
        -------
        Any
            The result of the callable


        Raises
        ------
        Timeout
            If the locking times out.
        """
        lock = SoftFileLock(lock_file, timeout=lock_timeout)
        with lock.acquire(timeout=lock_timeout):
            return callable()

    def _get_index_path(self, table: str) -> str:
        return os.path.join(self.cache_dir, f'{table}.json')

    def _get_lock_path(self, table: str) -> str:
        return f'{self._get_index_path(table)}.lock'

    def _get_db(self, table: str) -> TinyDB:
        if table == 'index':
            raise ValueError('The table must not be `index`!')
        serialization = SerializationMiddleware()
        serialization.register_serializer(DateTimeSerializer(), 'DateTime')
        return TinyDB(self._get_index_path(table), storage=serialization)

    def _update_meta(self, table: str, doc_ids: Iterable[int], fields: Union[Mapping, Callable[[Mapping], None]]):
        table = self._get_db(table)
        doc_id = table.update(fields=fields, doc_ids=doc_ids)
        return doc_id

    def _upsert_meta(self, table: str, params: Dict[str, Any], experiment_id: Optional[int] = None) -> List[int]:
        meta = {} if self.experiment is None else {'commit': self.experiment.mainfile.commit,
                                                   'is_dirty': self.experiment.mainfile.is_dirty,
                                                   'filename': os.path.basename(self.experiment.mainfile.filename)}
        data = {'params': params,
                'meta': meta,
                'time': datetime.utcnow(),
                'experiment_id': experiment_id}

        table = self._get_db(table)
        doc_id = table.upsert(data, Query().params == params)
        return doc_id

    def _remove_meta(self, table: str, params: Dict[str, Any] = None, doc_ids: Optional[Iterable[int]] = None,
                     sexperiment_id: Optional[int] = None) -> List[int]:
        query = None
        if params is not None:
            query = Query().params == params
        return self._get_db(table).remove(cond=query,
                                          doc_ids=doc_ids)

    def _find_meta_by_exact_params(self, table: str, params: Dict[str, Any],
                                   experiment_id: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._get_db(table).search(Query().params == params)

    def _find_meta(self, table: str, match_condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = Query()
        composite_condition = None
        for key, value in match_condition.items():
            current_condition = query['params'][key] == value
            composite_condition = (
                current_condition
                if composite_condition is None
                else composite_condition & current_condition
            )
        if composite_condition is None:
            return self._get_db(table).all()

        return self._get_db(table).search(composite_condition)

    def _build_artifact_path(self, artifact_type: str, id: Union[int, str]) -> str:
        path = os.path.join(self.cache_dir, artifact_type)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'{artifact_type}_{id}.pt')
        return path

    def save_artifact(self, artifact_type: str, params: Dict[str, Any], artifact: Dict[str, Any]) -> str:
        """Saves an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.
        artifact : Dict[str, Any]
            The actual artifact to be stored.

        Returns
        -------
        str
            File storage location.

        Raises
        ------
        RuntimeError
            In case more than one artifact with identical configuration is found.
        """
        ids = Storage.locked_call(
            lambda: self._upsert_meta(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )
        if len(ids) != 1:
            raise RuntimeError(f'The index contains duplicates (artifact_type={artifact_type}, params={params})')

        try:
            path = self._build_artifact_path(artifact_type, ids[0])
            torch.save(artifact, path)
            return path
        except:  # noqa: E722
            Storage.locked_call(
                lambda: self._remove_meta(artifact_type, params),
                self._get_lock_path(artifact_type),
                self.lock_timeout
            )
            raise

    def load_artifact(self, artifact_type: str, params: Dict[str, Any],
                      return_params: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Loads an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.
        return_params : bool, optional
            If True also the parameters are returned, by default False.

        Returns
        -------
        Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]
            The artifact and optionally the params.

        Raises
        ------
        RuntimeError
            In case more than one artifact with identical configuration is found.
        """
        documents = Storage.locked_call(
            lambda: self._find_meta_by_exact_params(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )
        if len(documents) == 0:
            return None
        elif len(documents) > 1:
            raise RuntimeError(f'The index contains duplicates (artifact_type={artifact_type}, params={params})')

        document = documents[0]
        path = self._build_artifact_path(artifact_type, document.doc_id)
        if return_params:
            return torch.load(path), document['params']
        else:
            return torch.load(path)

    def find_artifacts(self, artifact_type: str, match_condition: Dict[str, Any],
                       return_documents_only=False) -> List[Dict[str, Any]]:
        """Find all artifacts matching the defined parameters.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        match_condition : Dict[str, Any]
            Subset of parameters to query artifacts.
        return_artifact_id: bool, optional
            If True also the model id is returned, by default False.

        Returns
        -------
        List[Dict[str, Any]]
            List of loaded artifact params and artifacts (use key `artifact` to retrieve artifact).
        """
        raw_documents = Storage.locked_call(
            lambda: self._find_meta(artifact_type, match_condition),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        documents = []
        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            document['id'] = document_id

            if not return_documents_only:
                document['artifact'] = torch.load(self._build_artifact_path(artifact_type, document_id))
            documents.append(document)
        return documents

    def save_sparse_matrix(self,
                           artifact_type: str,
                           params: Dict[str, Any],
                           sparse_matrix: sp.csr_matrix,
                           ignore_duplicate: bool = False) -> str:
        """Saves a scipy matrix artifact

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.
        sparse_matrix : scipy.sparse.csr_matrix
            The actual artifact to be stored.
        ignore_duplicate : bool
            If True and another sparse_matrix is found with the same configuration,
            the new sparse_matrix will not be saved, by default False.

        Returns
        -------
        str
            File storage location.
        """
        ppr_idx = None
        if "ppr_idx" in params.keys() and not isinstance(params["ppr_idx"], int):
            ppr_idx = np.array(params["ppr_idx"])
            params["ppr_idx"] = hash(frozenset(params["ppr_idx"]))

        if ignore_duplicate:
            # check there's no entry with the exact same config already present
            ids = Storage.locked_call(
                lambda: self._find_meta_by_exact_params(artifact_type, params),
                self._get_lock_path(artifact_type),
                self.lock_timeout
            )
            if len(ids) > 0:
                logging.info("Ignoring duplicate save in save_sparse_matrix call")
                return self._build_artifact_path(artifact_type, ids[0].doc_id).replace(".pt", ".npz")

        ids = Storage.locked_call(
            lambda: self._upsert_meta(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )
        if len(ids) != 1:
            raise RuntimeError(f'The index contains duplicates (artifact_type={artifact_type}, params={params})')

        try:
            path = self._build_artifact_path(artifact_type, ids[0]).replace(".pt", ".npz")
            sp.save_npz(path, sparse_matrix)
            logging.info("Saved sparse matrix to storage")
            if ppr_idx is not None:
                ppr_path = path.replace(".npz", "idx.npy")
                np.save(ppr_path, ppr_idx)
                logging.info("Saved ppr index to storage")
            return path
        except:  # noqa: E722
            Storage.locked_call(
                lambda: self._remove_meta(artifact_type, params),
                self._get_lock_path(artifact_type),
                self.lock_timeout
            )
            raise

    def find_sparse_matrix(self,
                           artifact_type: str,
                           match_condition: Dict[str, Any],
                           find_first=False,
                           return_id: bool = False,
                           return_documents_only=False
                           ) -> List[Union[Tuple[sp.csr_matrix, Dict[str, Any]],
                                           Tuple[sp.csr_matrix, Dict[str, Any], int]]]:
        """Find all sparse matrices matching the defined parameters.

        Parameters
        ----------
        artifact_type: str
            Identifier of artifact type.
        match_condition: Dict[str, Any]
            Subset of parameters to query artifacts.
        find_first: bool, optional
            If True only the first match is returned, by default False.
        return_id: bool, optional
            If True also the sparse_tensors id is returned, by default False.

        Returns
        -------
        List[Union[Tuple[sp.csr_matrix, Dict[str, Any]], Tuple[sp.csr_matrix, Dict[str, Any], int]]]
            List of loaded matrices, their params and optionally the ids.
        """

        if "ppr_idx" in match_condition.keys() and not isinstance(match_condition["ppr_idx"], int):
            match_condition["ppr_idx"] = hash(frozenset(match_condition["ppr_idx"]))

        raw_documents = Storage.locked_call(
            lambda: self._find_meta(artifact_type, match_condition),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )
        # to get the most recent documents first we revert the list
        raw_documents.reverse()
        if return_documents_only:
            for document in raw_documents:
                if "ppr_idx" in document['params'].keys() and isinstance(document['params']["ppr_idx"], int):
                    path = self._build_artifact_path(artifact_type, document.doc_id).replace(".pt", ".npz")
                    ppr_path = path.replace(".npz", "idx.npy")
                    document['params']["ppr_idx"] = np.load(ppr_path)
            return raw_documents
        results = []
        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            path = self._build_artifact_path(artifact_type, document_id).replace(".pt", ".npz")
            sparse_matrix = sp.load_npz(path)

            if "ppr_idx" in document['params'].keys() and isinstance(document['params']["ppr_idx"], int):
                ppr_path = path.replace(".npz", "idx.npy")
                document['params']["ppr_idx"] = np.load(ppr_path)

            if return_id:
                results.append((sparse_matrix, document['params'], document_id))
            else:
                results.append((sparse_matrix, document['params']))

            if find_first:
                return results
        return results

    def hash_sparse_matrix(self,
                           artifact_type: str,
                           match_condition: Dict[str, Any]
                           ) -> List[Union[Tuple[sp.csr_matrix, Dict[str, Any]],
                                           Tuple[sp.csr_matrix, Dict[str, Any], int]]]:
        """Find all sparse matrices matching the defined parameters.

        Parameters
        ----------
        artifact_type: str
            Identifier of artifact type.
        match_condition: Dict[str, Any]
            Subset of parameters to query artifacts.
        Returns
        -------
        List[Union[Tuple[sp.csr_matrix, Dict[str, Any]], Tuple[sp.csr_matrix, Dict[str, Any], int]]]
            List of loaded matrices, their params and optionally the ids.
        """

        raw_documents = Storage.locked_call(
            lambda: self._find_meta(artifact_type, match_condition),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            if "ppr_idx" in document['params'].keys() and not isinstance(document['params']["ppr_idx"], int):

                path = self._build_artifact_path(artifact_type, document_id).replace(".pt", ".npz")
                ppr_path = path.replace(".npz", "idx.npy")
                ppr_idx = np.array(document['params']["ppr_idx"])

                document['params']["ppr_idx"] = hash(frozenset(ppr_idx))
                Storage.locked_call(
                    lambda: self._update_meta(artifact_type, doc_ids=[document_id], fields=document),
                    self._get_lock_path(artifact_type),
                    self.lock_timeout,
                )
                np.save(ppr_path, ppr_idx)

        return raw_documents

    def remove_sparse_matrices(self,
                               artifact_type: str,
                               match_condition: Dict[str, Any]
                               ) -> List[Union[Tuple[sp.csr_matrix, Dict[str, Any]],
                                               Tuple[sp.csr_matrix, Dict[str, Any], int]]]:
        """Find all sparse matrices matching the defined parameters.

        Parameters
        ----------
        artifact_type: str
            Identifier of artifact type.
        match_condition: Dict[str, Any]
            Subset of parameters to query artifacts.

        Returns
        -------
        List[Union[Tuple[sp.csr_matrix, Dict[str, Any]], Tuple[sp.csr_matrix, Dict[str, Any], int]]]
            List of loaded matrices, their params and optionally the ids.
        """

        raw_documents = Storage.locked_call(
            lambda: self._find_meta(artifact_type, match_condition),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        doc_ids = [document.doc_id for document in raw_documents]
        removed_docs = Storage.locked_call(
            lambda: self._remove_meta(artifact_type, doc_ids=doc_ids),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            path = self._build_artifact_path(artifact_type, document_id).replace(".pt", ".npz")
            try:
                os.remove(path)

                if "ppr_idx" in document['params'].keys() and isinstance(document['params']["ppr_idx"], int):
                    ppr_path = path.replace(".npz", "idx.npy")
                    os.remove(ppr_path)
            except OSError:
                pass

        return removed_docs

    def save_model(self, artifact_type: str, params: Dict[str, Any], model: MODEL_TYPE) -> str:
        """Saves an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.
        model : MODEL_TYPE
            The actual artifact to be stored.

        Returns
        -------
        str
            File storage location.
        """
        state_dict = model.cpu().state_dict()
        return self.save_artifact(artifact_type, params, state_dict)

    def load_model(self, artifact_type: str, params: Dict[str, Any]) -> MODEL_TYPE:
        """Loads an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.

        Returns
        -------
        MODEL_TYPE
            The artifact and optionally the params.

        Raises
        ------
        RuntimeError
            In case more than one artifact with identical configuration is found.
        """
        artifact, params = self.load_artifact(artifact_type, params, return_params=True)
        model = create_model(params)
        model.load_state_dict(artifact)
        return model

    def find_models(
        self, artifact_type: str, match_condition: Dict[str, Any], return_model_id: bool = False
    ) -> List[Union[Tuple[MODEL_TYPE, Dict[str, Any]], Tuple[MODEL_TYPE, Dict[str, Any], int]]]:
        """Find all models matching the defined parameters.

        Parameters
        ----------
        artifact_type: str
            Identifier of artifact type.
        match_condition: Dict[str, Any]
            Subset of parameters to query artifacts.
        return_model_id: bool, optional
            If True also the model id is returned, by default False.

        Returns
        -------
        List[Union[Tuple[MODEL_TYPE, Dict[str, Any]], Tuple[MODEL_TYPE, Dict[str, Any], int]]]
            List of loaded models, their params and optionally the ids.
        """
        documents = self.find_artifacts(artifact_type, match_condition)
        results = []
        for document in documents:
            model = create_model(document['params'])
            model.load_state_dict(document['artifact'])
            if return_model_id:
                results.append((model, document['params'], document['id']))
            else:
                results.append((model, document['params']))
        return results
