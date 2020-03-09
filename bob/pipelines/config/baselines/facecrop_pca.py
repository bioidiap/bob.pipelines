import bob.bio.face
import bob.bio.base
import functools

####
# PREPROCESSOR
#####

# Using face crop
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

## eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

preprocessor = functools.partial(
    bob.bio.face.preprocessor.FaceCrop,
    cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
)

#####
# Extractor
#####

extractor = bob.bio.base.extractor.Linearize

####
# Algorithm
##

from bob.pipelines.bob_bio.blocks import AlgorithmAdaptor
from bob.bio.base.algorithm import PCA

# Wrapping bob.bio.base algorithms with the bob.pipelines adaptor
algorithm = AlgorithmAdaptor(functools.partial(PCA, 0.99))
