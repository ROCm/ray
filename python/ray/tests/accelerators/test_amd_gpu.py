import os
import sys
from unittest.mock import patch

import pytest

import ray
from ray._private.accelerators import (
    AMDGPUAcceleratorManager,
    get_accelerator_manager_for_resource,
)
from ray._private.accelerators.amd_gpu import (
    HIP_VISIBLE_DEVICES_ENV_VAR,
    NOSET_HIP_VISIBLE_DEVICES_ENV_VAR,
)
from ray._private.accelerators.nvidia_gpu import CUDA_VISIBLE_DEVICES_ENV_VAR


@pytest.mark.parametrize(
    "visible_devices_env_var", ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
)
@patch(
    "ray._private.accelerators.AMDGPUAcceleratorManager.get_current_node_num_accelerators",  # noqa: E501
    return_value=4,
)
def test_visible_amd_gpu_ids(
    mock_get_num_accelerators, visible_devices_env_var, monkeypatch, shutdown_only
):
    monkeypatch.setenv(visible_devices_env_var, "0,1,2")
    # Delete the cache so it can be re-populated the next time
    # we call get_accelerator_manager_for_resource
    del get_accelerator_manager_for_resource._resource_name_to_accelerator_manager
    ray.init()
    _ = mock_get_num_accelerators.called
    assert ray.available_resources()["GPU"] == 3


@patch(
    "ray._private.accelerators.AMDGPUAcceleratorManager._get_amd_device_ids",
    return_value=["0x74a1", "0x74a1", "0x74a1", "0x74a1"],
)
def test_visible_amd_gpu_type(mock_get_amd_device_ids, shutdown_only):
    ray.init()
    _ = mock_get_amd_device_ids.called
    assert (
        AMDGPUAcceleratorManager.get_current_node_accelerator_type()
        == "AMD-Instinct-MI300X-OAM"
    )


@patch(
    "ray._private.accelerators.AMDGPUAcceleratorManager._get_amd_device_ids",
    return_value=["0x640f", "0x640f", "0x640f", "0x640f"],
)
def test_visible_amd_gpu_type_bad_device_id(mock_get_num_accelerators, shutdown_only):
    ray.init()
    _ = mock_get_num_accelerators.called
    assert AMDGPUAcceleratorManager.get_current_node_accelerator_type() is None


@pytest.mark.parametrize(
    "visible_devices_env_var", ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
)
def test_get_current_process_visible_accelerator_ids(
    visible_devices_env_var, monkeypatch
):
    monkeypatch.setenv(visible_devices_env_var, "0,1,2")
    assert AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() == [
        "0",
        "1",
        "2",
    ]

    monkeypatch.setenv(visible_devices_env_var, "0,2,7")
    assert AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() == [
        "0",
        "2",
        "7",
    ]

    monkeypatch.setenv(visible_devices_env_var, "")
    assert AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() == []

    del os.environ[visible_devices_env_var]
    assert (
        AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() is None
    )


def test_hip_cuda_env_var_get_current_process_visible_accelerator_ids(monkeypatch):
    # HIP and CUDA visible env vars are set and equal
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() == [
        "0",
        "1",
        "2",
    ]

    # HIP and CUDA visible env vars are set and not equal
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3")
    with pytest.raises(ValueError):
        AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids()


def test_set_current_process_visible_accelerator_ids():
    AMDGPUAcceleratorManager.set_current_process_visible_accelerator_ids(["0"])
    env_var = AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
    assert os.environ[env_var] == "0"

    AMDGPUAcceleratorManager.set_current_process_visible_accelerator_ids(["0", "1"])
    assert os.environ[env_var] == "0,1"

    AMDGPUAcceleratorManager.set_current_process_visible_accelerator_ids(
        ["0", "1", "7"]
    )
    assert os.environ[env_var] == "0,1,7"

    del os.environ[env_var]


def test_get_resource_name():
    assert AMDGPUAcceleratorManager.get_resource_name() == "GPU"


def test_get_visible_accelerator_ids_env_var_defaults_to_hip(monkeypatch):
    for key in (
        HIP_VISIBLE_DEVICES_ENV_VAR,
        CUDA_VISIBLE_DEVICES_ENV_VAR,
        "ROCR_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    assert (
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        == HIP_VISIBLE_DEVICES_ENV_VAR
    )


def test_get_visible_accelerator_ids_env_var_hip_only(monkeypatch):
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES_ENV_VAR, raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(HIP_VISIBLE_DEVICES_ENV_VAR, "0,1")
    assert (
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        == HIP_VISIBLE_DEVICES_ENV_VAR
    )


def test_get_visible_accelerator_ids_env_var_cuda_only(monkeypatch):
    monkeypatch.delenv(HIP_VISIBLE_DEVICES_ENV_VAR, raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES_ENV_VAR, "0,1")
    assert (
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        == CUDA_VISIBLE_DEVICES_ENV_VAR
    )


def test_get_visible_accelerator_ids_env_var_both_equal_uses_hip(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(HIP_VISIBLE_DEVICES_ENV_VAR, "0,1,2")
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES_ENV_VAR, "0,1,2")
    assert (
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        == HIP_VISIBLE_DEVICES_ENV_VAR
    )


def test_get_visible_accelerator_ids_env_var_mismatched_hip_cuda(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(HIP_VISIBLE_DEVICES_ENV_VAR, "0,1,2")
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES_ENV_VAR, "0,1,3")
    with pytest.raises(
        ValueError, match="Inconsistent values found. Please use either"
    ):
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()


def test_get_visible_accelerator_ids_env_var_rocr_without_hip_raises(monkeypatch):
    monkeypatch.delenv(HIP_VISIBLE_DEVICES_ENV_VAR, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="instead of ROCR_VISIBLE_DEVICES"):
        AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()


def test_get_current_process_visible_accelerator_ids_no_dev_files(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "NoDevFiles")
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES_ENV_VAR, raising=False)
    assert AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids() == []


def test_validate_resource_request_quantity():
    ok, msg = AMDGPUAcceleratorManager.validate_resource_request_quantity(2.0)
    assert ok is True and msg is None


@patch("ray._private.accelerators.AMDGPUAcceleratorManager._get_amd_device_ids")
def test_get_current_node_accelerator_type_no_device_ids(mock_get):
    mock_get.return_value = None
    assert AMDGPUAcceleratorManager.get_current_node_accelerator_type() is None


@patch("ray._private.accelerators.AMDGPUAcceleratorManager._get_amd_device_ids")
def test_get_current_node_accelerator_type_get_ids_raises(mock_get):
    mock_get.side_effect = OSError("smi")
    assert AMDGPUAcceleratorManager.get_current_node_accelerator_type() is None


def test_gpu_name_to_accelerator_type():
    assert (
        AMDGPUAcceleratorManager._gpu_name_to_accelerator_type("0x75a0")
        == "AMD-Instinct-MI350X-OAM"
    )
    assert AMDGPUAcceleratorManager._gpu_name_to_accelerator_type("0x9999") is None
    assert AMDGPUAcceleratorManager._gpu_name_to_accelerator_type(None) is None


def test_set_current_process_visible_accelerator_ids_noset_hip_env(monkeypatch):
    monkeypatch.setenv(HIP_VISIBLE_DEVICES_ENV_VAR, "0")
    monkeypatch.setenv(NOSET_HIP_VISIBLE_DEVICES_ENV_VAR, "1")
    before = os.environ[HIP_VISIBLE_DEVICES_ENV_VAR]
    AMDGPUAcceleratorManager.set_current_process_visible_accelerator_ids(["1", "2", "3"])
    assert os.environ[HIP_VISIBLE_DEVICES_ENV_VAR] == before

    monkeypatch.setenv(NOSET_HIP_VISIBLE_DEVICES_ENV_VAR, "0")
    AMDGPUAcceleratorManager.set_current_process_visible_accelerator_ids(["1", "2", "3"])
    assert os.environ[HIP_VISIBLE_DEVICES_ENV_VAR] == "1,2,3"
    del os.environ[HIP_VISIBLE_DEVICES_ENV_VAR]


@patch("ray._private.thirdparty.pyamdsmi.smi_get_device_count", return_value=0)
@patch("ray._private.thirdparty.pyamdsmi.smi_shutdown")
@patch("ray._private.thirdparty.pyamdsmi.smi_initialize")
def test_get_current_node_num_accelerators_pyamdsmi_ok(
    mock_smi_init, mock_smi_shutdown, mock_get_count
):
    assert AMDGPUAcceleratorManager.get_current_node_num_accelerators() == 0
    mock_smi_init.assert_called_once()
    mock_get_count.assert_called_once()


@patch(
    "ray._private.thirdparty.pyamdsmi.smi_initialize",
    side_effect=OSError("no rocm"),
)
@patch("ray._private.thirdparty.pyamdsmi.smi_shutdown")
def test_get_current_node_num_accelerators_pyamdsmi_fails(
    mock_smi_shutdown, mock_smi_init
):
    assert AMDGPUAcceleratorManager.get_current_node_num_accelerators() == 0


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
