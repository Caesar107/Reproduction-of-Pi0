"""Custom robot policy inputs and outputs for training and inference."""

import dataclasses

import einops
import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class CustomRobotInputs(transforms.DataTransformFn):
    """Transform custom robot observations into model inputs.
    
    This transform converts your robot's observation format to the format expected by the model.
    
    Your dataset contains:
    - observation.state: joint states (8 dims)
    - observation.end_effector_state: ee states (8 dims)
    - observation.images.top: top camera image (256x256)
    - observation.images.wrist: wrist camera image (256x256)
    - action: actions (7 dims)
    """
    
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, x: dict) -> dict:
        # Extract observations from your dataset format
        joint_state = x["observation.state"]  # (8,)
        ee_state = x["observation.end_effector_state"]  # (8,)
        
        # Concatenate joint state and ee state to form the full state
        # Total: 16 dimensions (8 joint + 8 ee)
        # Convert to numpy arrays if they're tensors
        if hasattr(joint_state, 'numpy'):
            joint_state = joint_state.numpy()
        if hasattr(ee_state, 'numpy'):
            ee_state = ee_state.numpy()
        
        state = np.concatenate([joint_state, ee_state], axis=-1)
        
        # Images - convert from CHW to HWC format (like ALOHA policy)
        # LeRobot stores images in CHW format, but ResizeImages expects HWC
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel]
            return einops.rearrange(img, "c h w -> h w c")
        
        top_image = convert_image(x["observation.images.top"])
        wrist_image = convert_image(x["observation.images.wrist"])
        
        # Get task/prompt
        task = x.get("task", "manipulation task")
        
        # Like Libero/DROID: we need 3 image slots (base + left_wrist + right_wrist)
        # We only have 2 cameras (top as base, wrist as left_wrist)
        # Create padding image with explicit shape to ensure memory is properly allocated
        padding_image = np.zeros(top_image.shape, dtype=top_image.dtype)
        
        inputs = {
            "state": state,  # (16,) - will be normalized
            "image": {
                "base_0_rgb": top_image,  # Our top camera as base
                "left_wrist_0_rgb": wrist_image,  # Our wrist camera
                "right_wrist_0_rgb": padding_image,  # Padded (no second wrist)
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # PI0 model: mask out padding images (False)
                # PI0_FAST model: don't mask padding images (True)
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "prompt": task,
        }
        
        # Actions are only available during training (key is "action" in our dataset)
        if "action" in x:
            inputs["actions"] = np.asarray(x["action"])
        
        return inputs


@dataclasses.dataclass(frozen=True)
class CustomRobotOutputs(transforms.DataTransformFn):
    """Transform model outputs back into robot action format.
    
    This is used during inference to convert the model's predicted actions
    back into the format your robot expects.
    """

    def __call__(self, x: dict) -> dict:
        # Model outputs 32-dim actions (padded), but we only need the first 7 dims
        # This matches how ALOHA extracts [:14] from 32-dim output
        actions = np.asarray(x["actions"][:, :7])
        
        return {
            "actions": actions,  # (action_horizon, 7)
        }
