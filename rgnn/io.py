from datetime import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from filelock import SoftFileLock
import sacred
from tinydb import Query, TinyDB
from tinydb_serialization import SerializationMiddleware, Serializer
import torch
from torch.sparse import FloatTensor

from rgnn.models import create_model, MODEL_TYPE


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

    def __init__(self, cache_dir: str = 'cache', experiment: sacred.Experiment = None, lock_timeout: int = 10):
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
        if table not in self.dbs:
            serialization = SerializationMiddleware()
            serialization.register_serializer(DateTimeSerializer(), 'DateTime')
            self.dbs[table] = TinyDB(self._get_index_path(table), storage=serialization)
        return self.dbs[table]

    def _upsert_meta(self, table: str, params: Dict[str, Any], experiment_id: Optional[int] = None) -> List[int]:
        meta = {} if self.experiment is None else {'commit': self.experiment.mainfile.commit,
                                                   'is_dirty': self.experiment.mainfile.is_dirty,
                                                   'filename': os.path.basename(self.experiment.mainfile.filename)}
        data = {'params': params,
                'meta': meta,
                'time': datetime.utcnow(),
                'experiment_id': experiment_id}

        return self._get_db(table).upsert(data, Query().params == params)

    def _remove_meta(self, table: str, params: Dict[str, Any], experiment_id: Optional[int] = None) -> List[int]:
        return self._get_db(table).remove(Query().params == params)

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
        except:
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
        documents = self._find_meta_by_exact_params(artifact_type, params)
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

    def find_artifacts(self, artifact_type: str, match_condition: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        raw_documents = self._find_meta(artifact_type, match_condition)

        documents = []
        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            document['id'] = document_id
            document['artifact'] = torch.load(self._build_artifact_path(artifact_type, document_id))
            documents.append(document)
        return documents

    def save_sparse_tensor(self, artifact_type: str, params: Dict[str, Any], sparse_tensor: FloatTensor) -> str:
        """Saves an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.
        sparse_tensor : FloatTensor
            The actual artifact to be stored.

        Returns
        -------
        str
            File storage location.
        """
        artifact = {'edge_idx': sparse_tensor.indices().cpu(),
                    'edge_weight': sparse_tensor.values().cpu(),
                    'shape': sparse_tensor.shape}
        return self.save_artifact(artifact_type, params, artifact)

    def load_sparse_tensor(self, artifact_type: str, params: Dict[str, Any]) -> FloatTensor:
        """Loads an artifact.

        Parameters
        ----------
        artifact_type : str
            Identifier of artifact type.
        params : Dict[str, Any]
            parameters identifying the artifacts provenance.

        Returns
        -------
        Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]
            The artifact and optionally the params.

        Raises
        ------
        RuntimeError
            In case more than one artifact with identical configuration is found.
        """
        artifact = self.load_artifact(artifact_type, params)
        if artifact is None:
            return None
        else:
            return torch.sparse.FloatTensor(
                artifact['edge_idx'],
                artifact['edge_weight'],
                tuple(artifact['shape'])
            ).coalesce()

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
