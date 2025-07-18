from collections.abc import Iterable
from typing import AbstractSet  # noqa: UP035

import dagster as dg
import pytest
from dagster import (
    AssetExecutionContext,
    AssetKey,
    AutoMaterializePolicy,
    DagsterInstance,
    Definitions,
    _check as check,
)
from dagster._core.definitions.external_asset import (
    create_external_asset_from_source_asset,
    external_assets_from_specs,
)


def test_external_asset_basic_creation() -> None:
    assets_def = next(
        iter(
            external_assets_from_specs(
                specs=[
                    dg.AssetSpec(
                        key="external_asset_one",
                        description="desc",
                        metadata={"user_metadata": "value"},
                        group_name="a_group",
                    )
                ]
            )
        )
    )
    assert isinstance(assets_def, dg.AssetsDefinition)

    expected_key = dg.AssetKey(["external_asset_one"])

    assert assets_def.key == expected_key
    # will work once https://github.com/dagster-io/dagster/pull/16755 merges
    # assert assets_def.descriptions_by_key[expected_key] == "desc"
    assert assets_def.metadata_by_key[expected_key]["user_metadata"] == "value"
    assert assets_def.group_names_by_key[expected_key] == "a_group"
    assert assets_def.descriptions_by_key[expected_key] == "desc"
    assert not assets_def.is_executable


def test_external_asset_tags_owners() -> None:
    assets_def = next(
        iter(
            external_assets_from_specs(
                specs=[
                    dg.AssetSpec(
                        key="external_asset_one",
                        tags={"foo": "bar", "baz": "qux"},
                        owners=["ben@dagsterlabs.com"],
                    )
                ]
            )
        )
    )
    assert isinstance(assets_def, dg.AssetsDefinition)
    expected_key = dg.AssetKey(["external_asset_one"])
    assert assets_def.tags_by_key[expected_key] == {"foo": "bar", "baz": "qux"}
    assert assets_def.owners_by_key[expected_key] == ["ben@dagsterlabs.com"]


def test_external_asset_with_hyphens() -> None:
    key = dg.AssetKey(["with-hyphen", "external_asset_one"])
    assets_def = next(
        iter(
            external_assets_from_specs(
                specs=[
                    dg.AssetSpec(
                        key=key,
                    )
                ]
            )
        )
    )
    assert isinstance(assets_def, dg.AssetsDefinition)
    assert assets_def.key == key


def test_multi_external_asset_basic_creation() -> None:
    for assets_def in external_assets_from_specs(
        specs=[
            dg.AssetSpec(
                key="external_asset_one",
                description="desc",
                metadata={"user_metadata": "value"},
                group_name="a_group",
            ),
            dg.AssetSpec(
                key=dg.AssetKey(["value", "another_spec"]),
                description="desc",
                metadata={"user_metadata": "value"},
                group_name="a_group",
            ),
        ]
    ):
        assert isinstance(assets_def, dg.AssetsDefinition)


def test_invalid_external_asset_creation() -> None:
    invalid_specs = [
        dg.AssetSpec("invalid_asset1", auto_materialize_policy=AutoMaterializePolicy.eager()),
        dg.AssetSpec("invalid_asset2", code_version="ksjdfljs"),
        dg.AssetSpec("invalid_asset2", skippable=True),
    ]

    for invalid_spec in invalid_specs:
        with pytest.raises(check.CheckError):
            external_assets_from_specs(specs=[invalid_spec])


def test_normal_asset_materializeable() -> None:
    @dg.asset
    def an_asset() -> None: ...

    assert an_asset.is_executable


def test_external_asset_creation_with_deps() -> None:
    asset_two = dg.AssetSpec("external_asset_two")
    assets_def = next(
        iter(
            external_assets_from_specs(
                [
                    dg.AssetSpec(
                        "external_asset_one",
                        deps=[asset_two.key],  # todo remove key when asset deps accepts it
                    )
                ]
            )
        )
    )
    assert isinstance(assets_def, dg.AssetsDefinition)

    expected_key = dg.AssetKey(["external_asset_one"])

    assert assets_def.key == expected_key
    assert assets_def.asset_deps[expected_key] == {
        dg.AssetKey(["external_asset_two"]),
    }


def test_how_source_assets_are_backwards_compatible() -> None:
    class DummyIOManager(dg.IOManager):
        def handle_output(self, context, obj) -> None:
            pass

        def load_input(self, context) -> str:
            return "hardcoded"

    source_asset = dg.SourceAsset(key="source_asset", io_manager_def=DummyIOManager())

    @dg.asset
    def an_asset(source_asset: str) -> str:
        return source_asset + "-computed"

    defs_with_source = dg.Definitions(assets=[source_asset, an_asset])

    instance = DagsterInstance.ephemeral()

    result_one = defs_with_source.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance
    )

    assert result_one.success
    assert result_one.output_for_node("an_asset") == "hardcoded-computed"

    defs_with_shim = dg.Definitions(
        assets=[create_external_asset_from_source_asset(source_asset), an_asset]
    )

    assert isinstance(defs_with_shim.resolve_assets_def("source_asset"), dg.AssetsDefinition)

    result_two = defs_with_shim.resolve_implicit_global_asset_job_def().execute_in_process(
        instance=instance,
        # currently we have to explicitly select the asset to exclude the source from execution
        asset_selection=[dg.AssetKey("an_asset")],
    )

    assert result_two.success
    assert result_two.output_for_node("an_asset") == "hardcoded-computed"


def get_job_for_assets(defs: Definitions, *coercibles_or_defs) -> dg.JobDefinition:
    job_def = defs.resolve_implicit_job_def_def_for_assets(
        set_from_coercibles_or_defs(coercibles_or_defs)
    )
    assert job_def, "Expected to find a job def"
    return job_def


def set_from_coercibles_or_defs(coercibles_or_defs: Iterable) -> AbstractSet["AssetKey"]:
    return set(
        [
            AssetKey.from_coercible_or_definition(coercible_or_def)
            for coercible_or_def in coercibles_or_defs
        ]
    )


def test_how_partitioned_source_assets_are_backwards_compatible() -> None:
    class DummyIOManager(dg.IOManager):
        def handle_output(self, context, obj) -> None:
            pass

        def load_input(self, context) -> str:
            return "hardcoded"

    partitions_def = dg.DailyPartitionsDefinition(start_date="2021-01-01")
    source_asset = dg.SourceAsset(
        key="source_asset", io_manager_def=DummyIOManager(), partitions_def=partitions_def
    )

    @dg.asset(partitions_def=partitions_def)
    def an_asset(context: AssetExecutionContext, source_asset: str) -> str:
        return source_asset + "-computed-" + context.partition_key

    assert an_asset.partitions_def is partitions_def
    assert source_asset.partitions_def is partitions_def

    defs_with_source = dg.Definitions(assets=[source_asset, an_asset])

    instance = DagsterInstance.ephemeral()

    job_def_without_shim = get_job_for_assets(defs_with_source, an_asset)

    result_one = job_def_without_shim.execute_in_process(
        instance=instance, partition_key="2021-01-02"
    )

    assert result_one.success
    assert result_one.output_for_node("an_asset") == "hardcoded-computed-2021-01-02"

    shimmed_source_asset = create_external_asset_from_source_asset(source_asset)
    defs_with_shim = dg.Definitions(assets=[shimmed_source_asset, an_asset])

    assert isinstance(defs_with_shim.resolve_assets_def("source_asset"), dg.AssetsDefinition)

    job_def_with_shim = get_job_for_assets(defs_with_shim, an_asset)

    result_two = job_def_with_shim.execute_in_process(
        instance=instance,
        # currently we have to explicitly select the asset to exclude the source from execution
        asset_selection=[dg.AssetKey("an_asset")],
        partition_key="2021-01-03",
    )

    assert result_two.success
    assert result_two.output_for_node("an_asset") == "hardcoded-computed-2021-01-03"


def test_observable_source_asset_decorator() -> None:
    freshness_policy = dg.LegacyFreshnessPolicy(maximum_lag_minutes=30)

    @dg.observable_source_asset(legacy_freshness_policy=freshness_policy)
    def an_observable_source_asset() -> dg.DataVersion:
        return dg.DataVersion("foo")

    assets_def = create_external_asset_from_source_asset(an_observable_source_asset)
    assert assets_def.is_executable
    assert assets_def.is_observable
    assert (
        assets_def.legacy_freshness_policies_by_key[an_observable_source_asset.key]
        == freshness_policy
    )
    defs = dg.Definitions(assets=[assets_def])

    instance = DagsterInstance.ephemeral()
    result = defs.resolve_implicit_global_asset_job_def().execute_in_process(instance=instance)

    assert result.success
    assert result.output_for_node("an_observable_source_asset") is None

    all_observations = result.get_asset_observation_events()
    assert len(all_observations) == 1
    observation_event = all_observations[0]
    assert observation_event.asset_observation_data.asset_observation.data_version == "foo"

    all_materializations = result.get_asset_materialization_events()
    assert len(all_materializations) == 0


def test_external_assets_with_dependencies_manual_construction() -> None:
    upstream_asset = dg.AssetSpec("upstream_asset")
    downstream_asset = dg.AssetSpec("downstream_asset", deps=[upstream_asset])

    @dg.multi_asset(name="_generated_asset_def_1", specs=[upstream_asset])
    def _upstream_def(context: AssetExecutionContext) -> None:
        raise Exception("do not execute")

    @dg.multi_asset(name="_generated_asset_def_2", specs=[downstream_asset])
    def _downstream_asset(context: AssetExecutionContext) -> None:
        raise Exception("do not execute")

    defs = dg.Definitions(assets=[_upstream_def, _downstream_asset])
    assert defs

    assert defs.resolve_implicit_global_asset_job_def().asset_layer.asset_graph.get(
        dg.AssetKey("downstream_asset")
    ).parent_keys == {dg.AssetKey("upstream_asset")}


def test_external_asset_multi_asset() -> None:
    upstream_asset = dg.AssetSpec("upstream_asset")
    downstream_asset = dg.AssetSpec("downstream_asset", deps=[upstream_asset])

    @dg.multi_asset(specs=[downstream_asset, upstream_asset])
    def _generated_asset_def(context: AssetExecutionContext):
        raise Exception("do not execute")

    defs = dg.Definitions(assets=[_generated_asset_def])
    assert defs

    assert defs.resolve_asset_graph().asset_dep_graph["upstream"][downstream_asset.key] == {
        upstream_asset.key
    }


def test_external_assets_with_dependencies() -> None:
    upstream_asset = dg.AssetSpec("upstream_asset")
    downstream_asset = dg.AssetSpec("downstream_asset", deps=[upstream_asset])

    defs = dg.Definitions(assets=external_assets_from_specs([upstream_asset, downstream_asset]))
    assert defs

    assert defs.resolve_asset_graph().asset_dep_graph["upstream"][downstream_asset.key] == {
        upstream_asset.key
    }
