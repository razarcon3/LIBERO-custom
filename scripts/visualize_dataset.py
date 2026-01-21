import h5py
import imageio


def main():
    example_dataset_file = "datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate/training_set.hdf5"

    with h5py.File(example_dataset_file, "r") as f:
        print(f)
        images = f["data/demo_0/obs/agentview_rgb"][()]

    video_writer = imageio.get_writer("output.mp4", fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()
    
if __name__ == "__main__":
    main()