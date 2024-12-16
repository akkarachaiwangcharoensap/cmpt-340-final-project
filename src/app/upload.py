import os
from src.app.file_exceptions import FileNotFoundError, InvalidFileTypeError, FileSaveError

def upload_file(file):
    """
    Handles the upload of a file, validates its type, and saves it to the ./tmp directory.

    :param file: A file object (from `request.files`) to be uploaded and processed.
    :type file: werkzeug.datastructures.FileStorage

    :raises FileNotFoundError: If no file is provided or the file has an empty filename.
    :raises InvalidFileTypeError: If the file does not have a `.dicom` extension.
    :raises FileSaveError: If the file cannot be saved due to an error.

    :return: None
    """
    # Define valid file extensions
    valid_extensions = ['.dcm', '.dcim', '.dc3', '.dic', '.dicom']

    # Validate the file presence
    if not file or file.filename == '':
        raise FileNotFoundError('No file part in the request or file is missing.')

    # Validate the file extension
    file_extension = os.path.splitext(file.filename)[1].lower()  # Extract the extension
    if file_extension not in valid_extensions:
        raise InvalidFileTypeError(
            f"Invalid file type. Allowed types are: {', '.join(valid_extensions)}"
        )

    # Ensure the ./tmp directory exists
    os.makedirs('./tmp', exist_ok=True)

    # Save the file to the ./tmp directory
    file_path = os.path.join('./tmp', file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        raise FileSaveError(f"Failed to save file: {e}")

    return file_path

def remove_file(file_name):
    """
    Removes a file from the ./tmp directory.

    :param file_name: Name of the file to remove
    :raises FileNotFoundError: If the file does not exist
    :raises FileSaveError: If the file removal fails

    :return: None
    """
    file_path = os.path.join('./tmp', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in the ./tmp directory")
    try:
        os.remove(file_path)
    except Exception as e:
        raise FileSaveError(f"Failed to remove file '{file_name}': {e}")