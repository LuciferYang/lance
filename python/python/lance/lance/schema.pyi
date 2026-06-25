# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Any, Dict, List, Optional

import pyarrow as pa

class LanceField:
    def name(self) -> str: ...
    def id(self) -> int: ...
    def children(self) -> List[LanceField]: ...
    def is_unenforced_primary_key(self) -> bool: ...
    def unenforced_primary_key_position(self) -> Optional[int]: ...
    def initial_default(self) -> Optional[Any]:
        """Return the decoded ``lance-schema:initial-default`` for this field.

        Returns a pyarrow scalar whose type matches the field's Arrow type, or
        ``None`` when the ``lance-schema:initial-default`` key is absent.

        To set or remove the default on a dataset column use
        :meth:`LanceDataset.set_column_default` or
        :meth:`LanceDataset.remove_column_default`.
        """
        ...

    def write_default(self) -> Optional[Any]:
        """Return the decoded ``lance-schema:write-default`` for this field.

        Returns a pyarrow scalar whose type matches the field's Arrow type, or
        ``None`` when the key is absent.  In Model A this key is never set, so
        this method always returns ``None`` in Model A datasets.

        Does **not** fall back to :meth:`initial_default`.  See
        :meth:`effective_default` for the combined fallback logic.
        """
        ...

    def effective_default(self) -> Optional[Any]:
        """Return the effective default for this field.

        Returns :meth:`write_default` when ``lance-schema:write-default`` is
        present, otherwise :meth:`initial_default`.  Returns ``None`` when
        neither key is set.
        """
        ...

class LanceSchema:
    def fields(self) -> List[LanceField]: ...
    def to_pyarrow(self) -> pa.Schema: ...
    @staticmethod
    def from_pyarrow(schema: pa.Schema) -> "LanceSchema": ...

def schema_to_json(schema: pa.Schema) -> Dict[str, Any]: ...
def json_to_schema(schema_json: Dict[str, Any]) -> pa.Schema: ...
