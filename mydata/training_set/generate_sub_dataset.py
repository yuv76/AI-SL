import os
import shutil


def copy_images_at_intervals(source_dir, destination_dir, interval):
    os.makedirs(destination_dir, exist_ok=True)

    # Get the list of all files in the source directory, sorted for consistency
    files = sorted(os.listdir(source_dir))

    # Filter for image files
    image_files = [f for f in files if f.lower().endswith(".png")]

    # Copy images at the specified interval
    for i in range(0, len(image_files), interval):
        src_path = os.path.join(source_dir, image_files[i])
        dest_path = os.path.join(destination_dir, source_dir[0] + image_files[i])
        shutil.copy2(src_path, dest_path)
        print(f"Copied {image_files[i]} to {destination_dir}")


if __name__ == "__main__":
    destination_directory = "#/"
    leap = 4

    copy_images_at_intervals("E/", destination_directory, leap)
    copy_images_at_intervals("M/", destination_directory, leap)
    copy_images_at_intervals("N/", destination_directory, leap)
    copy_images_at_intervals("S/", destination_directory, leap)
