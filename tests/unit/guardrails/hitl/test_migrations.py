import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql

from lisa.guardrails.hitl.models import Approval, AuditEvent, Claim, Execution, Revision

ROOT = Path(__file__).resolve().parents[4]
MODELS = (Execution, Approval, Revision, Claim, AuditEvent)


def load_revision():
    path = ROOT / "migrations/versions/a41c9d72e830_add_hitl_approvals.py"
    spec = importlib.util.spec_from_file_location("hitl_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_columns(table):
    return {tuple(column.name for column in constraint.columns)
            for constraint in table.constraints if isinstance(constraint, sa.UniqueConstraint)}


def test_revision_chain():
    scripts = ScriptDirectory(str(ROOT / "migrations"))
    assert scripts.get_heads() == ["b872d961a703"]
    assert set(scripts.get_revision("b872d961a703").down_revision) == {"a41c9d72e830", "7f4a89b99ea1"}
    assert scripts.get_revision("a41c9d72e830").down_revision == "f0069c95796f"
    assert load_revision().down_revision == "f0069c95796f"


def test_explicit_migration_matches_current_models(monkeypatch):
    migration = load_revision()
    buffer = StringIO()
    operations = Operations(MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer},
    ))
    monkeypatch.setattr(migration, "op", operations)
    tables = {}
    create_table = operations.create_table

    def capture(name, *args, **kwargs):
        tables[name] = create_table(name, *args, **kwargs)
        return tables[name]

    with patch.object(operations, "create_table", side_effect=capture), patch.object(
        operations, "create_index", wraps=operations.create_index,
    ) as create_index:
        migration.upgrade()
    assert set(tables) == {model.__tablename__ for model in MODELS}
    dialect = postgresql.dialect()
    for model in MODELS:
        actual, expected = tables[model.__tablename__], model.__table__
        assert list(actual.columns.keys()) == list(expected.columns.keys())
        for column in expected.columns:
            migrated = actual.c[column.name]
            assert migrated.nullable == column.nullable
            assert migrated.primary_key == column.primary_key
            assert str(migrated.type.compile(dialect=dialect)) == str(column.type.compile(dialect=dialect))
            assert {key.target_fullname for key in migrated.foreign_keys} == {
                key.target_fullname for key in column.foreign_keys
            }
        assert unique_columns(actual) == unique_columns(expected)
    actual_indexes = {
        call.args[0]: (call.args[1], tuple(call.args[2]), call.kwargs.get("unique", False),
                       str(call.kwargs.get("postgresql_where", "")))
        for call in create_index.call_args_list
    }
    expected_indexes = {
        index.name: (model.__tablename__, tuple(column.name for column in index.columns),
                     index.unique, str(predicate) if predicate is not None else "")
        for model in MODELS for index in model.__table__.indexes
        for predicate in [index.dialect_options["postgresql"].get("where")]
    }
    assert actual_indexes == expected_indexes
    sql = buffer.getvalue()
    assert "WHERE state IN ('pending', 'approved', 'claimed', 'executing', 'resuming')" in sql
    assert "CREATE FUNCTION hitl_reject_mutation() RETURNS trigger" in sql
    assert "RAISE EXCEPTION" in sql
    for table in ("hitl_events", "hitl_revisions"):
        assert f"BEFORE UPDATE OR DELETE ON {table}" in sql
        assert "FOR EACH ROW EXECUTE FUNCTION hitl_reject_mutation()" in sql


def test_offline_downgrade_removes_guards_before_tables(monkeypatch):
    migration = load_revision()
    buffer = StringIO()
    monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer},
    )))
    migration.downgrade()
    sql = buffer.getvalue()
    assert sql.index("DROP TRIGGER") < sql.index("DROP FUNCTION") < sql.index("DROP TABLE")
    assert sql.index("DROP TABLE hitl_claims") < sql.index("DROP TABLE hitl_approvals")
    assert sql.index("DROP TABLE hitl_revisions") < sql.index("DROP TABLE hitl_approvals")
    assert sql.index("DROP TABLE hitl_approvals") < sql.index("DROP TABLE hitl_executions")