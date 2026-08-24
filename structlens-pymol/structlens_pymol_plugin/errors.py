class BundleError(Exception):
    """Base error for invalid or incompatible StructLens bundles."""


class BundleCompatibilityError(BundleError):
    """The bundle schema major is newer than this plugin supports."""


class UnsafeBundleError(BundleError):
    """The archive contains unsafe paths or executable content."""
