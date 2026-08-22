class SelectorError(Exception):
    """Base selector error."""
class ConfigurationError(SelectorError): pass
class ManifestError(SelectorError): pass
class ArtifactError(SelectorError): pass
class InferenceError(SelectorError): pass
