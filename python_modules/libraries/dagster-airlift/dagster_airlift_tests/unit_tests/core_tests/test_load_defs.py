from pathlib import Path
from typing import cast
from unittest import mock

from dagster import (
    AssetKey,
    AssetsDefinition,
    AssetSpec,
    Definitions,
    asset,
    asset_check,
    executor,
    job,
    logger,
    multi_asset,
    schedule,
    sensor,
)
from dagster._core.code_pointer import CodePointer
from dagster._core.definitions.assets.definition.asset_dep import AssetDep
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.reconstruct import initialize_repository_def_from_pointer
from dagster._utils.test.definitions import (
    definitions,
    scoped_reconstruction_metadata,
    unwrap_reconstruction_metadata,
)
from dagster_airlift.constants import TASK_MAPPING_METADATA_KEY
from dagster_airlift.core import assets_with_task_mappings, build_defs_from_airflow_instance
from dagster_airlift.core.airflow_defs_data import AirflowDefinitionsData
from dagster_airlift.core.filter import AirflowFilter
from dagster_airlift.core.load_defs import (
    build_job_based_airflow_defs,
    enrich_airflow_mapped_assets,
    load_airflow_dag_asset_specs,
)
from dagster_airlift.core.multiple_tasks import assets_with_multiple_task_mappings
from dagster_airlift.core.serialization.compute import (
    build_airlift_metadata_mapping_info,
    compute_serialized_data,
)
from dagster_airlift.core.serialization.defs_construction import make_default_dag_asset_key
from dagster_airlift.core.serialization.serialized_data import (
    DagHandle,
    SerializedAirflowDefinitionsData,
    TaskHandle,
)
from dagster_airlift.core.utils import is_task_mapped_asset_spec, metadata_for_task_mapping
from dagster_airlift.test import asset_spec, make_instance
from dagster_shared.serdes import deserialize_value
from dagster_test.utils.definitions_execute_in_process import get_job_from_defs

from dagster_airlift_tests.unit_tests.conftest import (
    assert_dependency_structure_in_assets,
    fully_loaded_repo_from_airflow_asset_graph,
    load_definitions_airflow_asset_graph,
)


@executor  # pyright: ignore[reportCallIssue,reportArgumentType]
def nonstandard_executor(init_context):
    pass


@logger  # pyright: ignore[reportCallIssue,reportArgumentType]
def nonstandard_logger(init_context):
    pass


@sensor(job_name="the_job")
def some_sensor():
    pass


@schedule(cron_schedule="0 0 * * *", job_name="the_job")
def some_schedule():
    pass


@asset
def a():
    pass


b_spec = AssetSpec(key="b")


@asset_check(asset=a)  # pyright: ignore[reportArgumentType]
def a_check():
    pass


@job
def the_job():
    pass


def make_test_dag_asset_key(dag_id: str) -> AssetKey:
    return make_default_dag_asset_key("test_instance", dag_id)


def test_defs_passthrough() -> None:
    """Test that passed-through definitions are present in the final definitions."""
    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance({"dag": ["task"]}),
        defs=Definitions(
            assets=[a, b_spec],
            asset_checks=[a_check],
            jobs=[the_job],
            sensors=[some_sensor],
            schedules=[some_schedule],
            loggers={"the_logger": nonstandard_logger},  # pyright: ignore[reportArgumentType]
            executor=nonstandard_executor,  # pyright: ignore[reportArgumentType]
        ),
    )
    assert defs.executor == nonstandard_executor
    assert defs.loggers
    assert len(defs.loggers) == 1
    assert next(iter(defs.loggers.keys())) == "the_logger"
    assert defs.sensors
    assert len(list(defs.sensors)) == 2
    our_sensor = next(
        iter(sensor_def for sensor_def in defs.sensors if sensor_def.name == "some_sensor")
    )
    assert our_sensor == some_sensor
    assert defs.schedules
    assert len(list(defs.schedules)) == 1
    assert next(iter(defs.schedules)) == some_schedule
    assert defs.jobs
    assert len(list(defs.jobs)) == 1
    assert next(iter(defs.jobs)) == the_job
    repo = defs.get_repository_def()
    # Ensure that asset specs get properly coerced into asset defs
    assert set(repo.assets_defs_by_key.keys()) == {
        a.key,
        b_spec.key,
        make_test_dag_asset_key("dag"),
    }
    assert isinstance(repo.assets_defs_by_key[b_spec.key], AssetsDefinition)


def test_coerce_specs() -> None:
    """Test that asset specs are properly coerced into asset keys."""
    # Initialize an airflow instance with a dag "dag", which contains a task "task". There are no task instances or runs.

    spec = AssetSpec(key="a", metadata=metadata_for_task_mapping(task_id="task", dag_id="dag"))
    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance({"dag": ["task"]}),
        defs=Definitions(
            assets=[spec],
        ),
    )
    repo = defs.get_repository_def()
    assert len(repo.assets_defs_by_key) == 2
    assert AssetKey("a") in repo.assets_defs_by_key
    assets_def = repo.assets_defs_by_key[AssetKey("a")]
    # Asset metadata properties have been glommed onto the asset
    spec = next(iter(assets_def.specs))
    assert spec.metadata["Dag ID"] == "dag"


def test_invalid_dagster_named_tasks_and_dags() -> None:
    """Test that invalid dagster names are converted to valid names."""
    a = AssetKey("a")
    spec = AssetSpec(
        key=a,
        metadata=metadata_for_task_mapping(task_id="task-with-hyphens", dag_id="dag-with-hyphens"),
    )
    airflow_instance = make_instance({"dag-with-hyphens": ["task-with-hyphens"]})
    defs = build_defs_from_airflow_instance(
        airflow_instance=airflow_instance,
        defs=Definitions(
            assets=[spec],
        ),
    )

    repo = defs.get_repository_def()
    assert len(repo.assets_defs_by_key) == 2
    assert a in repo.assets_defs_by_key
    assets_def = repo.assets_defs_by_key[a]
    assert not assets_def.is_executable

    assert make_test_dag_asset_key("dag-with-hyphens") in repo.assets_defs_by_key
    dag_def = repo.assets_defs_by_key[
        make_default_dag_asset_key(airflow_instance.name, "dag_with_hyphens")
    ]
    assert not dag_def.is_executable


def has_single_task_handle(spec: AssetSpec, dag_id: str, task_id: str):
    assert len(spec.metadata[TASK_MAPPING_METADATA_KEY]) == 1
    task_handle_dict = next(iter(spec.metadata[TASK_MAPPING_METADATA_KEY]))
    return task_handle_dict["dag_id"] == dag_id and task_handle_dict["task_id"] == task_id


def test_transitive_asset_deps() -> None:
    """Test that cross-dag transitive asset dependencies are correctly generated."""
    # Asset graph is a -> b -> c where a and c are in different dags, and b isn't in any dag.
    repo_def = fully_loaded_repo_from_airflow_asset_graph(
        assets_per_task={
            "dag1": {"task": [("a", [])]},
            "dag2": {"task": [("c", ["b"])]},
        },
        additional_defs=Definitions(assets=[AssetSpec(key="b", deps=["a"])]),
    )
    repo_def.load_all_definitions()
    airflow_instance = make_instance(dag_and_task_structure={"dag1": ["task"], "dag2": ["task"]})
    dag1_key = make_default_dag_asset_key(instance_name=airflow_instance.name, dag_id="dag1")
    dag2_key = make_default_dag_asset_key(instance_name=airflow_instance.name, dag_id="dag2")
    a_key = AssetKey(["a"])
    b_key = AssetKey(["b"])
    c_key = AssetKey(["c"])
    assert len(repo_def.assets_defs_by_key) == 5
    assert set(repo_def.assets_defs_by_key.keys()) == {
        dag1_key,
        dag2_key,
        a_key,
        b_key,
        c_key,
    }

    dag1_asset = repo_def.assets_defs_by_key[dag1_key]
    assert [dep.asset_key for dep in next(iter(dag1_asset.specs)).deps] == [a_key]

    dag2_asset = repo_def.assets_defs_by_key[dag2_key]
    assert [dep.asset_key for dep in next(iter(dag2_asset.specs)).deps] == [c_key]

    a_asset = repo_def.assets_defs_by_key[a_key]
    assert [dep.asset_key for dep in next(iter(a_asset.specs)).deps] == []
    assert has_single_task_handle(next(iter(a_asset.specs)), "dag1", "task")

    b_asset = repo_def.assets_defs_by_key[b_key]
    assert [dep.asset_key for dep in next(iter(b_asset.specs)).deps] == [a_key]
    assert not is_task_mapped_asset_spec(next(iter(b_asset.specs)))

    c_asset = repo_def.assets_defs_by_key[c_key]
    assert [dep.asset_key for dep in next(iter(c_asset.specs)).deps] == [b_key]
    assert has_single_task_handle(next(iter(c_asset.specs)), "dag2", "task")


def test_peered_dags() -> None:
    """Test peered dags show up, and that linkage is preserved downstream of dags."""
    defs = load_definitions_airflow_asset_graph(
        assets_per_task={
            "dag1": {"task": []},
            "dag2": {"task": []},
            "dag3": {"task": []},
        },
        additional_defs=Definitions(
            assets=[AssetSpec(key="a", deps=[make_test_dag_asset_key("dag1")])]
        ),
    )
    assert defs.assets
    repo_def = defs.get_repository_def()
    repo_def.load_all_definitions()
    assert len(repo_def.assets_defs_by_key) == 4
    assert_dependency_structure_in_assets(
        repo_def=repo_def,
        expected_deps={
            make_test_dag_asset_key("dag1").to_user_string(): [],
            make_test_dag_asset_key("dag2").to_user_string(): [],
            make_test_dag_asset_key("dag3").to_user_string(): [],
            "a": [make_test_dag_asset_key("dag1").to_user_string()],
        },
    )
    for dag_asset_key in [
        make_test_dag_asset_key("dag1"),
        make_test_dag_asset_key("dag2"),
        make_test_dag_asset_key("dag3"),
    ]:
        dag_asset_spec = repo_def.assets_defs_by_key[dag_asset_key].specs_by_key[dag_asset_key]
        assert "dagster/kind/airflow" in dag_asset_spec.tags


def test_observed_assets() -> None:
    """Test that observed assets are properly linked to dags."""
    # Asset graph structure:
    #   a
    #  / \
    # b   c
    #  \ /
    #   d
    #  / \
    # e   f
    defs = load_definitions_airflow_asset_graph(
        assets_per_task={
            "dag": {
                "task1": [("a", []), ("b", ["a"]), ("c", ["a"])],
                "task2": [("d", ["b", "c"]), ("e", ["d"]), ("f", ["d"])],
            },
        },
    )
    assert defs.assets
    repo_def = defs.get_repository_def()
    repo_def.load_all_definitions()
    repo_def.load_all_definitions()
    assert len(repo_def.assets_defs_by_key) == 7
    assert_dependency_structure_in_assets(
        repo_def=repo_def,
        expected_deps={
            "a": [],
            "b": ["a"],
            "c": ["a"],
            "d": ["b", "c"],
            "e": ["d"],
            "f": ["d"],
            # Only leaf assets should be immediately upstream of the dag
            make_test_dag_asset_key("dag").to_user_string(): ["e", "f"],
        },
    )
    for key_str in ["a", "b", "c", "d", "e", "f"]:
        asset_spec = repo_def.assets_defs_by_key[AssetKey(key_str)].specs_by_key[AssetKey(key_str)]
        assert "dagster/kind/airliftmapped" in asset_spec.tags


def test_local_airflow_instance() -> None:
    """Test that a local-backed airflow instance can be correctly peered, and errors when the correct info can't be found."""
    defs = load_definitions_airflow_asset_graph(
        assets_per_task={
            "dag": {"task": [("a", [])]},
        },
        create_assets_defs=True,
    )

    assert defs.assets
    repo_def = defs.get_repository_def()

    defs = load_definitions_airflow_asset_graph(
        assets_per_task={
            "dag": {"task": [("a", [])]},
        },
        create_assets_defs=True,
    )
    repo_def = defs.get_repository_def()
    assert defs.assets
    repo_def = defs.get_repository_def()
    assert len(repo_def.assets_defs_by_key) == 2


@definitions
def airflow_instance_defs() -> Definitions:
    a = AssetKey("a")
    spec = AssetSpec(
        key=a,
        metadata=metadata_for_task_mapping(task_id="task", dag_id="dag"),
    )
    instance = make_instance({"dag": ["task"]})
    passed_in_defs = Definitions(assets=[spec])

    return build_defs_from_airflow_instance(airflow_instance=instance, defs=passed_in_defs)


def test_cached_loading() -> None:
    repository_def = initialize_repository_def_from_pointer(
        CodePointer.from_python_file(str(Path(__file__)), "airflow_instance_defs", None),
    )
    assert repository_def.repository_load_data
    assert len(repository_def.repository_load_data.reconstruction_metadata) == 1
    assert (
        "dagster-airlift/source/test_instance"
        in repository_def.repository_load_data.reconstruction_metadata
    )
    assert isinstance(
        repository_def.repository_load_data.reconstruction_metadata[
            "dagster-airlift/source/test_instance"
        ].value,
        str,
    )
    assert isinstance(
        deserialize_value(
            repository_def.repository_load_data.reconstruction_metadata[
                "dagster-airlift/source/test_instance"
            ].value
        ),
        SerializedAirflowDefinitionsData,
    )

    with scoped_reconstruction_metadata(unwrap_reconstruction_metadata(repository_def)):
        with mock.patch(
            "dagster_airlift.core.serialization.compute.compute_serialized_data",
            wraps=compute_serialized_data,
        ) as mock_compute_serialized_data:
            reloaded_repo_def = initialize_repository_def_from_pointer(
                CodePointer.from_python_file(str(Path(__file__)), "airflow_instance_defs", None),
            )
            assert mock_compute_serialized_data.call_count == 0
            assert reloaded_repo_def.assets_defs_by_key
            assert len(list(reloaded_repo_def.assets_defs_by_key.keys())) == 2
            assert {
                key
                for assets_def in reloaded_repo_def.assets_defs_by_key.values()
                for key in cast("AssetsDefinition", assets_def).keys
            } == {AssetKey("a"), make_test_dag_asset_key("dag")}


def test_multiple_tasks_per_asset(init_load_context: None) -> None:
    """Test behavior for a single AssetsDefinition where different specs map to different airflow tasks/dags."""

    @multi_asset(
        specs=[
            AssetSpec(key="a", metadata=metadata_for_task_mapping(task_id="task1", dag_id="dag1")),
            AssetSpec(key="b", metadata=metadata_for_task_mapping(task_id="task2", dag_id="dag2")),
        ],
        name="multi_asset",
    )
    def my_asset():
        pass

    instance = make_instance({"dag1": ["task1"], "dag2": ["task2"]})
    defs = build_defs_from_airflow_instance(
        airflow_instance=instance,
        defs=Definitions(assets=[my_asset]),
    )
    assert defs.assets
    # 3 Full assets definitions, but 4 keys
    assert len(list(defs.assets)) == 3
    assert {
        key for assets_def in defs.assets for key in cast("AssetsDefinition", assets_def).keys
    } == {
        AssetKey("a"),
        AssetKey("b"),
        make_test_dag_asset_key("dag1"),
        make_test_dag_asset_key("dag2"),
    }
    repo_def = defs.get_repository_def()
    a_and_b_asset = repo_def.assets_defs_by_key[AssetKey("a")]
    a_spec = next(iter(spec for spec in a_and_b_asset.specs if spec.key == AssetKey("a")))
    assert has_single_task_handle(a_spec, "dag1", "task1")
    b_spec = next(iter(spec for spec in a_and_b_asset.specs if spec.key == AssetKey("b")))
    assert has_single_task_handle(b_spec, "dag2", "task2")


def test_multiple_tasks_to_single_asset_metadata() -> None:
    instance = make_instance({"dag1": ["task1"], "dag2": ["task2"]})

    @asset(
        metadata={
            TASK_MAPPING_METADATA_KEY: [
                {"dag_id": "dag1", "task_id": "task1"},
                {"dag_id": "dag2", "task_id": "task2"},
            ]
        }
    )
    def an_asset() -> None: ...

    defs = build_defs_from_airflow_instance(
        airflow_instance=instance, defs=Definitions(assets=[an_asset])
    )

    assert defs.assets
    assert len(list(defs.assets)) == 3  # two dags and one asset

    assert defs.resolve_asset_graph().assets_def_for_key(AssetKey("an_asset")).specs_by_key[
        AssetKey("an_asset")
    ].metadata[TASK_MAPPING_METADATA_KEY] == [
        {"dag_id": "dag1", "task_id": "task1"},
        {"dag_id": "dag2", "task_id": "task2"},
    ]


def test_multiple_tasks_with_multiple_task_mappings() -> None:
    @asset
    def other_asset() -> None: ...

    @asset(deps=[other_asset])
    def scheduled_twice() -> None: ...

    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance(
            {"weekly_dag": ["task1"], "daily_dag": ["task1"], "other_dag": ["task1"]}
        ),
        defs=Definitions(
            assets=[
                other_asset,
                *assets_with_multiple_task_mappings(
                    assets=[scheduled_twice],
                    task_handles=[
                        {"dag_id": "weekly_dag", "task_id": "task1"},
                        {"dag_id": "daily_dag", "task_id": "task1"},
                    ],
                ),
            ],
        ),
    )

    Definitions.validate_loadable(defs)


def test_mixed_multiple_tasks_using_task_mappings() -> None:
    @asset
    def single_targeted_asset() -> None: ...

    @asset
    def double_targeted_asset() -> None: ...

    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance(
            {"weekly_dag": ["task1"], "daily_dag": ["task1"], "other_dag": ["task1"]}
        ),
        defs=Definitions.merge(
            Definitions(
                assets=assets_with_task_mappings(
                    dag_id="other_dag", task_mappings={"task1": [single_targeted_asset]}
                )
            ),
            Definitions(
                assets=assets_with_multiple_task_mappings(
                    assets=[double_targeted_asset],
                    task_handles=[
                        {"dag_id": "weekly_dag", "task_id": "task1"},
                        {"dag_id": "daily_dag", "task_id": "task1"},
                    ],
                )
            ),
        ),
    )

    Definitions.validate_loadable(defs)

    mapping_info = build_airlift_metadata_mapping_info(defs.assets)  # type: ignore
    assert mapping_info.all_mapped_asset_keys_by_dag_id["other_dag"] == {
        AssetKey("single_targeted_asset"),
    }
    assert mapping_info.all_mapped_asset_keys_by_dag_id["weekly_dag"] == {
        AssetKey("double_targeted_asset"),
    }
    assert mapping_info.all_mapped_asset_keys_by_dag_id["daily_dag"] == {
        AssetKey("double_targeted_asset"),
    }

    assert mapping_info.task_handle_map[AssetKey("single_targeted_asset")] == {
        TaskHandle(dag_id="other_dag", task_id="task1")
    }
    assert mapping_info.task_handle_map[AssetKey("double_targeted_asset")] == {
        TaskHandle(dag_id="weekly_dag", task_id="task1"),
        TaskHandle(dag_id="daily_dag", task_id="task1"),
    }


def test_task_mappings_in_same_dags() -> None:
    @asset
    def other_asset() -> None: ...

    @asset
    def double_targeted_asset() -> None: ...

    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance(
            {
                "weekly_dag": ["task1", "task_for_other_asset"],
                "daily_dag": ["task1"],
            }
        ),
        defs=Definitions.merge(
            Definitions(
                assets=[
                    *assets_with_multiple_task_mappings(
                        assets=[double_targeted_asset],
                        task_handles=[
                            {"dag_id": "weekly_dag", "task_id": "task1"},
                            {"dag_id": "daily_dag", "task_id": "task1"},
                        ],
                    )
                ]
            ),
            Definitions(
                assets=assets_with_task_mappings(
                    dag_id="weekly_dag",
                    task_mappings={"task_for_other_asset": [other_asset]},
                ),
            ),
        ),
    )

    Definitions.validate_loadable(defs)

    mapping_info = build_airlift_metadata_mapping_info(defs.assets)  # type: ignore
    assert mapping_info.all_mapped_asset_keys_by_dag_id["weekly_dag"] == {
        AssetKey("other_asset"),
        AssetKey("double_targeted_asset"),
    }
    assert mapping_info.all_mapped_asset_keys_by_dag_id["daily_dag"] == {
        AssetKey("double_targeted_asset"),
    }

    assert mapping_info.task_handle_map[AssetKey("other_asset")] == {
        TaskHandle(dag_id="weekly_dag", task_id="task_for_other_asset")
    }
    assert mapping_info.task_handle_map[AssetKey("double_targeted_asset")] == {
        TaskHandle(dag_id="weekly_dag", task_id="task1"),
        TaskHandle(dag_id="daily_dag", task_id="task1"),
    }


def test_task_mappings_with_same_task_id() -> None:
    @asset
    def other_asset() -> None: ...

    @asset
    def double_targeted_asset() -> None: ...

    defs = build_defs_from_airflow_instance(
        airflow_instance=make_instance(
            {
                "weekly_dag": ["task1"],
                "daily_dag": ["task1"],
            }
        ),
        defs=Definitions.merge(
            Definitions(
                assets=assets_with_multiple_task_mappings(
                    assets=[double_targeted_asset],
                    task_handles=[
                        {"dag_id": "weekly_dag", "task_id": "task1"},
                        {"dag_id": "daily_dag", "task_id": "task1"},
                    ],
                )
            ),
            Definitions(
                assets=assets_with_task_mappings(
                    dag_id="weekly_dag",
                    task_mappings={"task1": [other_asset]},
                ),
            ),
        ),
    )

    Definitions.validate_loadable(defs)

    mapping_info = build_airlift_metadata_mapping_info(defs.assets)  # type: ignore
    assert mapping_info.all_mapped_asset_keys_by_dag_id["weekly_dag"] == {
        AssetKey("other_asset"),
        AssetKey("double_targeted_asset"),
    }
    assert mapping_info.all_mapped_asset_keys_by_dag_id["daily_dag"] == {
        AssetKey("double_targeted_asset"),
    }

    assert mapping_info.task_handle_map[AssetKey("other_asset")] == {
        TaskHandle(dag_id="weekly_dag", task_id="task1")
    }
    assert mapping_info.task_handle_map[AssetKey("double_targeted_asset")] == {
        TaskHandle(dag_id="weekly_dag", task_id="task1"),
        TaskHandle(dag_id="daily_dag", task_id="task1"),
    }


def test_double_instance() -> None:
    airflow_instance_one = make_instance(
        dag_and_task_structure={"dag1": ["task1"]},
        instance_name="instance_one",
    )

    airflow_instance_two = make_instance(
        dag_and_task_structure={"dag1": ["task1"]},
        instance_name="instance_two",
    )

    defs_one = build_defs_from_airflow_instance(airflow_instance=airflow_instance_one)
    defs_two = build_defs_from_airflow_instance(airflow_instance=airflow_instance_two)

    defs = Definitions.merge(defs_one, defs_two)

    all_specs = {spec.key: spec for spec in defs.resolve_all_asset_specs()}

    assert set(all_specs.keys()) == {
        make_default_dag_asset_key("instance_one", "dag1"),
        make_default_dag_asset_key("instance_two", "dag1"),
    }


def test_enrich() -> None:
    spec = AssetSpec(key="a", metadata=metadata_for_task_mapping(task_id="task", dag_id="dag"))
    airflow_assets = enrich_airflow_mapped_assets(
        airflow_instance=make_instance({"dag": ["task"]}),
        mapped_assets=[spec],
        source_code_retrieval_enabled=None,
    )
    assert len(airflow_assets) == 1
    spec = next(iter(airflow_assets))
    assert isinstance(spec, AssetSpec)
    assert spec.key == AssetKey("a")
    # Asset metadata properties have been glommed onto the asset
    assert spec.metadata["Dag ID"] == "dag"


def test_load_dags() -> None:
    dag_assets = load_airflow_dag_asset_specs(
        airflow_instance=make_instance({"dag": ["task"]}),
    )
    assert len(dag_assets) == 1
    dag_asset = next(iter(dag_assets))
    assert dag_asset.key == make_default_dag_asset_key("test_instance", "dag")


def test_load_dags_upstream() -> None:
    upstream_task_spec = AssetSpec(
        key="a", metadata=metadata_for_task_mapping(task_id="task", dag_id="dag")
    )
    dag_assets = load_airflow_dag_asset_specs(
        airflow_instance=make_instance({"dag": ["task"]}),
        mapped_assets=[upstream_task_spec],
    )
    assert len(dag_assets) == 1
    dag_asset_spec = next(iter(dag_assets))
    assert dag_asset_spec.key == make_default_dag_asset_key("test_instance", "dag")
    assert len(list(dag_asset_spec.deps)) == 1
    assert next(iter(dag_asset_spec.deps)).asset_key == AssetKey("a")


def test_filtering() -> None:
    """Test using the retrieval filter to include/exclude dags."""
    instance = make_instance({"include_dag": ["task"], "exclude_dag": ["task"]})
    dag_assets = load_airflow_dag_asset_specs(
        airflow_instance=instance,
        retrieval_filter=AirflowFilter(dag_id_ilike="include"),
    )
    assert len(dag_assets) == 1
    # test tag based retrieval
    instance = make_instance(
        {"dag1": ["task"], "dag2": ["task"]},
        dag_props={"dag1": {"tags": ["first"]}, "dag2": {"tags": ["first", "second"]}},
    )
    dag_assets = load_airflow_dag_asset_specs(
        airflow_instance=instance,
        retrieval_filter=AirflowFilter(airflow_tags=["first"]),
    )
    assert len(dag_assets) == 2
    dag_assets = load_airflow_dag_asset_specs(
        airflow_instance=instance,
        retrieval_filter=AirflowFilter(airflow_tags=["first", "second"]),
    )
    assert len(dag_assets) == 1


def test_load_datasets() -> None:
    """Test automatic loading of datasets."""
    task_structure = {
        "producer1": ["producing_task"],
        "producer2": ["producing_task"],
        "consumer1": ["task"],
        "consumer2": ["task"],
    }
    # Dataset is produced and consumed by multiple tasks
    dataset_info = [
        {
            "uri": "s3://dataset-bucket/example1.csv",
            "producing_tasks": [
                {"dag_id": "producer1", "task_id": "producing_task"},
                {"dag_id": "producer2", "task_id": "producing_task"},
            ],
            "consuming_dags": ["consumer1", "consumer2"],
        },
        {
            "uri": "s3://dataset-bucket/example2.csv",
            "producing_tasks": [
                {"dag_id": "consumer1", "task_id": "task"},
            ],
            "consuming_dags": [],
        },
    ]
    af_instance = make_instance(
        dag_and_task_structure=task_structure,
        dataset_construction_info=dataset_info,
    )
    # Add an additional spec to the same task as the dataset
    spec = AssetSpec(
        key="a", metadata=metadata_for_task_mapping(task_id="producing_task", dag_id="producer1")
    )

    defs = build_defs_from_airflow_instance(
        airflow_instance=af_instance,
        defs=Definitions(assets=[spec]),
    )
    Definitions.validate_loadable(defs)

    definitions_data = AirflowDefinitionsData(
        airflow_instance=af_instance,
        resolved_repository=defs.get_repository_def(),
    )
    assert definitions_data.mapped_asset_keys_by_task_handle == {
        TaskHandle(dag_id="producer1", task_id="producing_task"): {
            AssetKey("example1"),
            AssetKey("a"),
        },
        TaskHandle(dag_id="producer2", task_id="producing_task"): {AssetKey("example1")},
        TaskHandle(dag_id="consumer1", task_id="task"): {AssetKey("example2")},
    }
    example1_spec = asset_spec("example1", defs)
    assert example1_spec
    assert example1_spec.deps == []
    example2_spec = asset_spec("example2", defs)
    assert example2_spec
    assert example2_spec.deps == [AssetDep("example1")]

    # Filter down to just producer1. Only example1 should be included
    defs = build_defs_from_airflow_instance(
        airflow_instance=af_instance,
        retrieval_filter=AirflowFilter(dag_id_ilike="producer1"),
        defs=Definitions(assets=[spec]),
    )
    Definitions.validate_loadable(defs)
    assert asset_spec("example1", defs)
    assert not asset_spec("example2", defs)


def test_load_job_defs() -> None:
    """Test job-based loader."""
    task_structure = {
        "producer1": ["producing_task"],
        "producer2": ["producing_task"],
        "consumer1": ["task"],
        "consumer2": ["task"],
    }
    # Dataset is produced and consumed by multiple tasks
    dataset_info = [
        {
            "uri": "s3://dataset-bucket/example1.csv",
            "producing_tasks": [
                {"dag_id": "producer1", "task_id": "producing_task"},
                {"dag_id": "producer2", "task_id": "producing_task"},
            ],
            "consuming_dags": ["consumer1", "consumer2"],
        },
        {
            "uri": "s3://dataset-bucket/example2.csv",
            "producing_tasks": [
                {"dag_id": "consumer1", "task_id": "task"},
            ],
            "consuming_dags": [],
        },
    ]
    af_instance = make_instance(
        dag_and_task_structure=task_structure,
        dataset_construction_info=dataset_info,
    )
    # Add an additional spec to the same task as the dataset
    spec = AssetSpec(
        key="a", metadata=metadata_for_task_mapping(task_id="producing_task", dag_id="producer1")
    )

    # Add an additional materializable asset to the same task
    @asset(metadata=metadata_for_task_mapping(task_id="producing_task", dag_id="producer1"))
    def b():
        pass

    defs = build_job_based_airflow_defs(
        airflow_instance=af_instance,
        mapped_defs=Definitions(assets=[spec, b]),
    )
    Definitions.validate_loadable(defs)
    assert isinstance(get_job_from_defs("producer1", defs), JobDefinition)
    assert isinstance(get_job_from_defs("producer2", defs), JobDefinition)
    assert isinstance(get_job_from_defs("consumer1", defs), JobDefinition)
    assert isinstance(get_job_from_defs("consumer2", defs), JobDefinition)

    airflow_defs_data = AirflowDefinitionsData(
        airflow_instance=af_instance,
        resolved_repository=defs.get_repository_def(),
    )

    repo = defs.get_repository_def()

    assert airflow_defs_data.airflow_mapped_jobs_by_dag_handle == {
        DagHandle(dag_id="producer1"): repo.get_job("producer1"),
        DagHandle(dag_id="producer2"): repo.get_job("producer2"),
        DagHandle(dag_id="consumer1"): repo.get_job("consumer1"),
        DagHandle(dag_id="consumer2"): repo.get_job("consumer2"),
    }
    assert airflow_defs_data.assets_per_job == {
        "producer1": {AssetKey("example1"), AssetKey("a"), AssetKey("b")},
        "producer2": {AssetKey("example1")},
        "consumer1": {AssetKey("example2")},
        "consumer2": set(),
    }
