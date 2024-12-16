class FileUploadError(Exception):
    """Base class for file upload errors."""
    pass

class FileNotFoundError(FileUploadError):
    """Raised when the file is not found in the request."""
    pass

class InvalidFileTypeError(FileUploadError):
    """Raised when the uploaded file has an invalid type."""
    pass

class FileSaveError(FileUploadError):
    """Raised when there is an error saving the file."""
    pass