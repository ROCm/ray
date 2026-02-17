import logging
import os
from typing import List, Optional, Tuple

from ray._private.accelerators.accelerator import AcceleratorManager
from ray._private.accelerators.nvidia_gpu import CUDA_VISIBLE_DEVICES_ENV_VAR

logger = logging.getLogger(__name__)

ROCR_VISIBLE_DEVICES_ENV_VAR = "ROCR_VISIBLE_DEVICES"
NOSET_ROCR_VISIBLE_DEVICES_ENV_VAR = "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"


class AMDGPUAcceleratorManager(AcceleratorManager):
    """AMD GPU accelerators."""

    @staticmethod
    def get_resource_name() -> str:
        return "GPU"

    @staticmethod
    def get_visible_accelerator_ids_env_var() -> str:
        #TODO: may need to also check cuda visible devices
        if(os.environ.get(ROCR_VISIBLE_DEVICES_ENV_VAR) == None && os.environ.get("HIP_VISIBLE_DEVICES") != None):
            os.environ[ROCR_VISIBLE_DEVICES_ENV_VAR] = os.environ["HIP_VISIBLE_DEVICES"]

        return ROCR_VISIBLE_DEVICES_ENV_VAR

    @staticmethod
    def get_current_process_visible_accelerator_ids() -> Optional[List[str]]:
        amd_visible_devices = os.environ.get(
            AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var(), None
        )

        if amd_visible_devices is None:
            return None

        if amd_visible_devices == "":
            return []

        if amd_visible_devices == "NoDevFiles":
            return []

        return list(amd_visible_devices.split(","))

    @staticmethod
    def get_current_node_num_accelerators() -> int:
        import amdsmi

        num_gpus = 0

        try:
            amdsmi.amdsmi_init()
        except amdsmi.AmdSmiLibraryException as e:
            #TODO: see if this message can be logged:
            #print(f"Failed to initialize AMD SMI library: {e}")
            return 0

        try:
            devices = amdsmi.amdsmi_get_processor_handles()
            num_gpus = len(devices)

        except amdsmi.AmdSmiException as e:
            #TODO: see if this message can be logged:
            #print(f"An error occurred while getting device handles: {e}")
            return 0

        finally:
            amdsmi.amdsmi_shut_down()

        return num_gpus

    @staticmethod
    def get_current_node_accelerator_type() -> Optional[str]:
        import amdsmi

        market_name = None
        try:
            amdsmi.amdsmi_init()
        except amdsmi.AmdSmiLibraryException as e:
            #TODO: see if this message can be logged:
            #print(f"Failed to initialize AMD SMI library: {e}")
            return None

        try:
            devices = amdsmi.amdsmi_get_processor_handles()
            num_gpus = len(devices)

        except amdsmi.AmdSmiException as e:
            #TODO: see if this message can be logged:
            #print(f"An error occurred while getting device handles: {e}")
            return None

        try:
            asic_info = amdsmi.amdsmi_get_gpu_asic_info(devices[0])
            market_name = asic_info['market_name']
        except amdsmi.AmdSmiException as e:
            #TODO: log e
            return None

        finally:
            amdsmi.amdsmi_shut_down()
        
        return market_name


    @staticmethod
    def validate_resource_request_quantity(
        quantity: float,
    ) -> Tuple[bool, Optional[str]]:
        return (True, None)

    @staticmethod
    def set_current_process_visible_accelerator_ids(
        visible_amd_devices: List[str],
    ) -> None:
        if os.environ.get(NOSET_ROCR_VISIBLE_DEVICES_ENV_VAR):
            return

        os.environ[
            AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()
        ] = ",".join([str(i) for i in visible_amd_devices])

