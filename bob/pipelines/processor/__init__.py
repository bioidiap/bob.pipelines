# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path

from .processor import ProcessorBlock, ProcessorPipeline

__path__ = extend_path(__path__, __name__)
