import bob.bio.face

# TODO: Fetch from bob.rc
original_directory = "/idiap/resource/database/mobio/IMAGES_PNG"
original_extension = ".png"
annotation_directory = "/idiap/resource/database/mobio/IMAGE_ANNOTATIONS"

# Using the bob.db package
bob_db = bob.bio.face.database.MobioBioDatabase(
    original_directory=original_directory,
    original_extension=original_extension,
    annotation_directory=annotation_directory,
)

# Wrapping it with the bob.pipelines adaptor
from bob.pipelines.bob_bio.annotated_blocks import (
    DatabaseConnectorAnnotated as DatabaseConnector,
)

database = DatabaseConnector(bob_db, protocol="mobile0-male")
