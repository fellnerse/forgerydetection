import click

from forgery_detection.data.file_lists import FileList


@click.command()
@click.option("--file_list_path", required=True, type=click.Path(exists=True))
@click.option("--new_root", required=True, type=click.Path(exists=True))
def change_root(file_list_path, new_root):
    f = FileList.load(file_list_path)
    print(f"Old root: {f.root}")

    f.root = str(new_root)
    print(f"New root: {f.root}")

    save_filelist = (
        click.prompt(
            "Is this root correct and save file list?", type=str, default="y"
        ).lower()
        == "y"
    )
    if save_filelist:
        f.save(file_list_path)
        print("Succesfully saved.")
    else:
        print("Aborted.")


if __name__ == "__main__":
    change_root()
