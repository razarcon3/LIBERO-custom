import argparse
import h5py
import numpy as np
import json

import libero.libero.utils.utils as libero_utils

from libero.libero.envs import *
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")
    parser.add_argument("--start-demo-num", type=int, default=0)
    args = parser.parse_args()

    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")

    env_kwargs = json.loads(f["data"].attrs["env_info"])
    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    all_demos = list(f["data"].keys())
    if args.start_demo_num >= len(all_demos):
        print(f"[Error] start_demo_num ({args.start_demo_num}) is out of bounds. Total demos: {len(all_demos)}")
        f.close()
        return
    demos = all_demos[args.start_demo_num:]
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )

    is_quitting = False

    def key_cb(key):
        nonlocal is_quitting
        if key != -1:
            is_quitting = True

    for ep in demos:
        if is_quitting:
            break
        print(f"Playing back episode {ep}... (press any key to quit)")

        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        reset_success = False
        while not reset_success:
            try:
                env.reset()
                reset_success = True
            except:
                continue

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        env.reset_from_xml_string(model_xml)
        env.viewer.set_camera(0)
        env.viewer.add_keypress_callback(key_cb)

        # load the flattened mujoco states and actions
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        for action in tqdm(actions):
            if is_quitting:
                break
            env.step(action)
            env.render()

    env.close()
    f.close()

if __name__ == "__main__":
    main()